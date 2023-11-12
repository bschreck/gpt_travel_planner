#!/bin/bash
GOOGLE_REGION=us-west1
SERVICE_ACCOUNT="cloud-run-deployer@travel-planner-404820.iam.gserviceaccount.com"
FUNCTION_NAME="get-flight-data"

gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --runtime=python312 \
    --region=$GOOGLE_REGION \
    --source=. \
    --entry-point=download_flight_data \
    --service-account="${SERVICE_ACCOUNT}" \
    --trigger-http \
    --timeout=1200s \
    --allow-unauthenticated \
    --set-secrets 'AVIATIONSTACK_API_KEY=AVIATIONSTACK_API_KEY:latest' \
    --memory=4Gi

FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME --region "$GOOGLE_REGION" --format='value(serviceConfig.uri)')

gcloud functions add-invoker-policy-binding "${FUNCTION_NAME}" \
    --region "$GOOGLE_REGION" \
    --member=serviceAccount:"${SERVICE_ACCOUNT}"

deleteOldSchedulerJob() {
    gcloud scheduler jobs delete $FUNCTION_NAME \
        --location=$GOOGLE_REGION \
        --quiet
}
deleteOldSchedulerJob || true

gcloud scheduler jobs create http $FUNCTION_NAME \
    --schedule="0 0 * * *" \
    --uri="${FUNCTION_URL}" \
    --http-method=GET \
    --time-zone="America/Los_Angeles" \
    --location=$GOOGLE_REGION \
    --oidc-service-account-email="${SERVICE_ACCOUNT}" \
