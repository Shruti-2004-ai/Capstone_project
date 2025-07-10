#!/bin/bash
# Test the full application workflow

# 1. Check requirements
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__}')"
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"

# 2. Run the app with test image
streamlit run rice_app.py --server.headless true --server.runOnSave true &
APP_PID=$!
sleep 5  # Wait for app to start

# 3. Verify CSV logging
if [ -f "predictions.csv" ]; then
    echo "✅ Logging system working"
    wc -l predictions.csv
else
    echo "❌ Logging system failed"
    exit 1
fi

kill $APP_PID
