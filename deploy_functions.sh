#!/bin/bash
#
GOOGLE_REGION=us-west1
SERVICE_ACCOUNT="cloud-run-deployer@travel-planner-404820.iam.gserviceaccount.com"
FUNCTION_NAME="gpt-travel-planner"

gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --runtime=python312 \
    --region=$GOOGLE_REGION \
    --service-account="${SERVICE_ACCOUNT}" \
    --source=. \
    --entry-point=functions_entrypoint \
    --timeout=600s \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars="FLIGHT_RADAR_EMAIL=$FLIGHT_RADAR_EMAIL" \
    --set-secrets 'AVIATIONSTACK_API_KEY=AVIATIONSTACK_API_KEY:latest','OPENAI_API_KEY=OPENAI_API_KEY:latest','DUFFEL_API_KEY=DUFFEL_API_KEY:latest','FLIGHT_RADAR_PASSWORD=FLIGHT_RADAR_PASSWORD:latest' \
    --memory=8Gi
