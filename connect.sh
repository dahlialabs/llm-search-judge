#!/bin/bash

export STAG_CLUSTER="gke_dahlia-infra-stag_us-central1_dahlia-stag-us-central1-gke"
export PROD_CLUSTER="gke_dahlia-infra-prod_us-central1_dahlia-prod-us-central1-gke"

kubectl --cluster="$STAG_CLUSTER" --namespace=daydream port-forward pods/inference-server-7977dd6444-5z9q5 10001:8080 &
KUBECTL_INF_STAG_PID=$!

# Proxy staging weaviate
# HTTP - 8080 -> local 8081
# GRPC - 50051 -> local 50052
kubectl --cluster="$STAG_CLUSTER" --namespace=daydream port-forward pods/eck-elasticsearch-es-default-0 9201:9200 &
KUBECTL_ES_STAG_PID=$!

kubectl --cluster="$PROD_CLUSTER" --namespace=daydream port-forward pods/eck-elasticsearch-es-default-0 9202:9200 &
KUBECTL_ES_PROD_PID=$!

export ELASTICSEARCH_STAG_USER=$(kubectl --cluster="$STAG_CLUSTER" get secret -n daydream elasticsearch-basic-auth-app-user -oyaml | yq '.data | map_values(@base64d) | .username' )
export ELASTICSEARCH_STAG_PASSWORD=$(kubectl --cluster="$STAG_CLUSTER" get secret -n daydream elasticsearch-basic-auth-app-user -oyaml | yq '.data | map_values(@base64d) | .password' )

export ELASTICSEARCH_PROD_USER=$(kubectl --cluster="$PROD_CLUSTER" get secret -n daydream elasticsearch-basic-auth-app-user -oyaml | yq '.data | map_values(@base64d) | .username' )
export ELASTICSEARCH_PROD_PASSWORD=$(kubectl --cluster="$PROD_CLUSTER" get secret -n daydream elasticsearch-basic-auth-app-user -oyaml | yq '.data | map_values(@base64d) | .password' )

function cleanup {
    kill $KUBECTL_ES_STAG_PID
    kill $KUBECTL_ES_PROD_PID
    kill $KUBECTL_INF_STAG_PID

    # Clear trap
    trap - SIGINT SIGTERM EXIT

}

trap cleanup SIGINT SIGTERM EXIT
