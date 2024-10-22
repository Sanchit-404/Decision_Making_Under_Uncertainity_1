#!/bin/bash

# Directory containing the .py files (change this to the correct path if needed)
FOLDER_PATH="./"

# Loop through all .py files in the folder and execute them
for file in "$FOLDER_PATH"*.py; do
    if [ -f "$file" ]; then
        echo "Executing $file..."
        python3 "$file"  # or use python3 depending on your environment
        if [ $? -ne 0 ]; then
            echo "Execution of $file failed!"
        else
            echo "$file executed successfully."
        fi
    fi
done

