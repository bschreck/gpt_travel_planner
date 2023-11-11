#!/bin/bash

gcloud functions deploy python-http-function \
    --gen2 \
    --runtime=python312 \
    --region=us-west1 \
    --source=. \
    --entry-point=pick_flights \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars="FLIGHT_RADAR_EMAIL=$FLIGHT_RADAR_EMAIL"
