#!/bin/bash

DATE_FORMAT="+%Y-%m-%d"

# Loop through all Python files in the repo
for file in $(find . -type f -name "*.py"); do
    if [[ -f "$file" ]]; then
        # Get Created date from first Git commit of the file
        CREATED_DATE=$(git log --diff-filter=A --follow --format=%as -- "$file" | tail -1)

        # If not in Git history, assume today's date
        if [[ -z "$CREATED_DATE" ]]; then
            CREATED_DATE=$(date "$DATE_FORMAT")
        fi

        # Update the Created and Last Modified tags in the file
        sed -i "s/^# Created:.*/# Created: $CREATED_DATE/" "$file"
        sed -i "s/^# Last Modified:.*/# Last Modified: $(date "$DATE_FORMAT")/" "$file"
    fi
done

echo "Updated Created and Last Modified dates for all Python files."
