import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.disable(logging.WARNING)

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import csv
import json
import os.path

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from blinkpy.blinkpy import Blink
from datetime import datetime


def display_image(img):
    cv2.imshow("Bird Watcher", img)
    key = cv2.waitKey(30000)
    if key == 27:
        cv2.destroyAllWindows()
        return False
    return True


def interpret_result(results, species_to_skip):
    found = {}
    for species_code, confidence in results:
        if confidence > confidence_threshold:
            if species_code_mappings[species_code] in scientific_common_name_mappings:
                species_common_name = scientific_common_name_mappings[species_code_mappings[species_code]]
                if species_common_name.lower() not in (to_skip.lower() for to_skip in species_to_skip):
                    found[species_common_name] = confidence

    return found


def export_to_db(species, location, camera_name, api, bucket, org):
    try:
        now = datetime.utcnow()
        for species_name, confidence in species.items():
            data_point = Point("sighting").tag("location", location).tag("camera", camera_name) \
                .tag("species", species_name).field("confidence", float(confidence)).time(now, WritePrecision.NS)
            api.write(bucket, org, data_point)
    except BaseException as err:
        print("An error occurred while attempting to write to InfluxDB: " + format(err)
              + " Metrics will be skipped for this image.")


def create_gui(species, image):
    bg_color = (0, 0, 0)
    bg = np.full(image.shape, bg_color, dtype=np.uint8)

    starting_y_coordinate = 30
    for species_name, confidence in sorted(species.items(), key=lambda x: x[1], reverse=True):
        cv2.putText(bg, species_name + ': ' + str(confidence), (5, starting_y_coordinate),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        starting_y_coordinate += 30

    x, y, w, h = cv2.boundingRect(bg[:, :, 2])
    display_result = image.copy()
    display_result[y:y + h, x:x + w] = bg[y:y + h, x:x + w]
    return display_result


def initialize_tensorflow_graph():
    # Create our inference graph
    img_placeholder = tf.placeholder(tf.string)
    decoded_img = tf.image.decode_jpeg(img_placeholder)
    decoded_image_float = tf.image.convert_image_dtype(
        image=decoded_img, dtype=tf.float32
    )

    # Expanding image from (height, width, 3) to (1, height, width, 3)
    image_tensor = tf.expand_dims(decoded_image_float, 0)

    # Load the model from tfhub.dev, and create a detector_output tensor
    model_url = "https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"
    detector = hub.Module(model_url)
    detector_result = detector(image_tensor, as_dict=True)

    # Initialize the Session
    init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]
    new_session = tf.Session()
    new_session.run(init_ops)
    return new_session, decoded_img, detector_result, img_placeholder


# We're using tensorflow v1 functionality here.
tf.disable_v2_behavior()
tf.enable_eager_execution()

# Ensure necessary configuration files are found and initialized.
config_path = "config.json"
while not os.path.isfile(config_path):
    configPath = input("Config not found. Enter the configuration file path. If you wish not to be prompted again, "
                       "drop the file (config.json) in this program's directory.")

bird_label_file = 'bird_labels.csv'
bird_common_to_scientific_file = 'bird_common_to_scientific.csv'

config = json.load(open("config.json"))
influx_config = config["influx_db"]
confidence_threshold = float(config["confidence_threshold"])
exportToInfluxDB = influx_config["enabled"]

# Init the DB connection.
if exportToInfluxDB:
    try:
        client = InfluxDBClient(url=influx_config["url"], token=influx_config["token"])
        write_api = client.write_api(write_options=SYNCHRONOUS)
    except BaseException as error:
        print("An error occurred while establishing an InfluxDB client:" + format(error)
              + " InfluxDB connection will be disabled.")
        exportToInfluxDB = False

# Check that necessary files exist.
if not os.path.isfile(bird_label_file) or not os.path.isfile(bird_common_to_scientific_file):
    print("The mapping files for bird species (" + bird_label_file + ") and common to scientific name mappings (" +
          bird_common_to_scientific_file + ") are missing from the currently executing directory. Ensure the files are "
                                           "present and retry.")
    raise SystemExit(0)

# Initialize species name dictionary.
reader = csv.DictReader(open('bird_labels.csv'))
species_code_mappings = {}
for row in reader:
    species_code_mappings[int(row['id'])] = row['name'].lower()

# Initialize scientific to common name dictionary.
reader = csv.DictReader(open('bird_common_to_scientific.csv'))
scientific_common_name_mappings = {}
for row in reader:
    scientific_common_name_mappings[row['scientific'].lower()] = row['common']

# Initialize camera feed.
print("You will be prompted to enter your Blink login information:")
blink = Blink()
blink.start()
camera = blink.cameras[config["blink_camera_name"]]

with tf.Graph().as_default():
    sess, detector_output, decoded_image, image_string_placeholder = initialize_tensorflow_graph()

    while True:
        # Load our image from the Blink camera.
        pic = camera.snap_picture()
        blink.refresh()
        raw_image = cv2.imdecode(np.frombuffer(camera.get_media().raw.data, np.uint8), -1)
        resized_image = cv2.resize(raw_image, (224, 224))
        image_bytes = cv2.imencode('.jpg', resized_image)[1].tobytes()

        # Run the graph.
        img_out, result = sess.run([detector_output, decoded_image], feed_dict={image_string_placeholder: image_bytes})

        # Interpret the findings.
        species_found = interpret_result(enumerate(result["default"][0]), config["species_to_ignore"])

        # Create the GUI portion.
        image_to_display = create_gui(species_found, raw_image)

        # Enter into InfluxDB.
        if exportToInfluxDB:
            export_to_db(species_found, config["location_name"], config["blink_camera_name"], write_api,
                         influx_config["org"], influx_config["bucket"])

        # Display GUI / latest image. Allow exit with ESC key.
        if not display_image(image_to_display):
            break
