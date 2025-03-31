#!/bin/bash

echo "Starting job on $(date)"

echo "Working directory: $(pwd)"

# Initialize your conda or nvm environment here, if needed

# Run Frontend
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
cd ./annotation_tool/Annotation-Interface
nvm install 16.20.2
nvm alias default 16.20.2
node -v
npm start > npm.log 2>&1 &
frontend_pid=$!  

# Run Backend
conda activate ikea-video-backend
cd ./annotation_tool/Annotation-Interface/backend
python app.py > backend.log 2>&1 &
backend_pid=$!

# Wait for both processes to complete
wait $frontend_pid $backend_pid

echo "Job Completed on $(date)"
exit 0