#!/bin/bash

# Stop running containers based on your image
docker ps -q --filter ancestor=ktrizzo/photorch-app | xargs -r docker stop

# Remove any stopped containers with the name "photorch-app"
docker rm photorch-app 2>/dev/null || true

# Run the container in the background (detached mode)
docker run --name photorch-app -p 8501:8501 -d ktrizzo/photorch-app

# Wait a few seconds to give the app time to start
sleep 1

# Open the default browser to the app URL (macOS, Linux, Windows)
if which open > /dev/null; then
  # macOS
  open http://localhost:8501
elif which xdg-open > /dev/null; then
  # Linux
  xdg-open http://localhost:8501
elif which start > /dev/null; then
  # Windows (Git Bash)
  start http://localhost:8501
else
  echo "Please open your browser and navigate to http://localhost:8501"
fi