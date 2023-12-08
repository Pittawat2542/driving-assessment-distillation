#!/bin/bash

# Change directory
input_dir="dev_routes"

cd "${LEADERBOARD_ROOT}/${input_dir}" || exit

# Log file to track executed files
LOG_FILE="${LEADERBOARD_ROOT}/executed_files.log"

# Loop through all XML files in the directory
for file in *.xml; do
    # Extract the base name without extension
    filename=$(basename "$file" .xml)
    OBJ_ID="OBJ_${filename}"
    export OBJ_ID

    # Get the current timestamp
    current_time=$(date +"%Y-%m-%d %H:%M:%S")

    # Log the processed file with timestamp
    echo "[$current_time] Processing file: $file" >> "$LOG_FILE"
    echo "[$current_time] Processing file: $file"

    # Use OBJ_ID in subsequent commands
    echo "OBJ_ID for $file is: $OBJ_ID"

    # Set the timeout for each iteration to 200 seconds
    timeout 240s bash -c "
        export ROUTES='${LEADERBOARD_ROOT}/${input_dir}/${file}'
        export REPETITIONS=1
        export DEBUG_CHALLENGE=1
        export TEAM_AGENT='${LEADERBOARD_ROOT}/leaderboard/autoagents/npc_obj_collection.py'
        export CHALLENGE_TRACK_CODENAME=SENSORS
        export CHECKPOINT_ENDPOINT=${LEADERBOARD_ROOT}/results.json
        ${LEADERBOARD_ROOT}/scripts/run_evaluation.sh  # Execute the evaluation script
    "
done
