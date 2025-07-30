import webbrowser
import time
import threading
from app import app

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    # Start browser opening in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start Flask app
    print("Starting Flask ML Trainer...")
    print("The application will open in your browser automatically.")
    print("If it doesn't open, go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=False, host='localhost', port=5000)
