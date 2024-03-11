#!/bin/bash

# Find all processes using port 5000 and store them in an array
processes=($(lsof -ti tcp:5000))

# Check if the array is empty
if [ ${#processes[@]} -eq 0 ]; then
    echo "No processes found on port 5000."
    exit 0
fi

# Print out the processes
echo "Processes using port 5000:"
for pid in "${processes[@]}"; do
    echo "PID: $pid"
done

# Ask user for confirmation
read -p "Do you want to delete all these processes? (yes/no) " answer
if [ "$answer" != "yes" ]; then
    echo "Exiting without killing any processes."
    exit 0
fi

# Kill the processes
for pid in "${processes[@]}"; do
    kill -9 $pid
    echo "Killed process $pid."
done

echo "All processes using port 5000 have been terminated."
