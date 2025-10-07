#!/bin/bash

for i in {1..10}
do
    echo "Running iteration $i"
    python -m src.main --config $1
    echo "Completed iteration $i"
done

echo "All iterations completed!"