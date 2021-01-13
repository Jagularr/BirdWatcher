# Bird Watcher
This program uses Blink security cameras to take pictures every 30 seconds and identify any birds in the frame.
Bird Watcher uses Google's pre-trained `aiy/vision/classifier/birds_V1` model.

### Usage tips:
- Set your camera as close as possible to the area where birds are expected (likely a feeder).
- The closer and more well-lit the bird, the higher likelihood of an accurate classification.
- Use the *species_to_ignore* and *confidence_threshold* config properties to tailor the sensitivity of classification.
- Should be good for identifying most North American birds.
- The model is intended for a well-cropped image of a bird so a quality and close image is important.
- The model is very fast, but can often make mistakes. Confirm any out of the ordinary sightings with your eyes.

### Config Description:
- *location_name* - This will be used to tag the location name in Influx DB.
- *species_to_ignore* - Use any number of species common names to ignore when classifying. This can be useful if the 
  model has a tendency to continuously incorrectly identify the same species, perhaps due to the background or some object in the
  frame. This will cause each species listed to be ignored when it is identified and it will not be reported.
- *confidence_threshold* - Sets how confident the model must be in a sighting before it can be reported. Setting this
lower than ~.25 will likely lead to a lot of mis-identifications.
- *influx_db* - Can be enabled/disabled. This object contains connection properties for an Influx DB v2 instance.
- *blink_camera_name* - The name of the blink camera to use. Note that currently only one camera can be monitored.

### Sample Config:
```
{
  "location_name": "David's Backyard",
  "species_to_ignore": ["Ring-Necked Pheasant"],
  "confidence_threshold": 0.30,
  "influx_db": {
    "enabled": true,
    "url": "localhost:8086",
    "token": "ABCDEFGHIJKLMNOPQRSTUVWXYQ123456789",
    "org": "Bird Enthusiasts",
    "bucket": "Bird Watching"
  },
  "blink_camera_name": "Bird Camera"
}
```

### TODO:
- Multiple camera support.
- Configurable camera interval, as it is hardcoded at 30 seconds. Blink's API might rate-limit faster requests so
it might not be feasible to be any faster.
