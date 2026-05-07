#!/bin/bash

echo "Building TimeSeries Plotter Docker Image..."
docker build -t timeseries-plotter:latest .

echo "Image built successfully!"
echo "To run the container:"
echo "  docker-compose up -d"
echo ""
echo "Or manually:"
echo "  docker run -d -p 5000:5000 -v \$(pwd)/uploads:/app/uploads -v \$(pwd)/plots:/app/plots timeseries-plotter:latest"