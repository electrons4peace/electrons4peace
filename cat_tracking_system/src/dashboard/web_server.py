"""
Web dashboard for cat tracking system
"""
import os
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import base64
import cv2
import numpy as np

# Use Flask for web server
from flask import Flask, render_template, jsonify, send_from_directory, request


class WebDashboard:
    """
    Web dashboard for monitoring the cat tracking system.
    Provides a simple web interface for viewing detected cats and statistics.
    """
    
    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        """
        Initialize the web dashboard
        
        Args:
            port: Port number for the web server
            host: Host address for the web server
        """
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.host = host
        self.app = Flask(
            __name__, 
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static")
        )
        
        # Ensure template and static directories exist
        os.makedirs(self.app.template_folder, exist_ok=True)
        os.makedirs(self.app.static_folder, exist_ok=True)
        
        # Create default templates if not exist
        self._create_default_templates()
        
        # Initialize data storage
        self.data = {
            "cats": {},            # Current cat detections
            "history": [],         # Detection history
            "system_status": {     # System status
                "running": False,
                "uptime": 0,
                "start_time": time.time(),
                "frame_count": 0,
                "detection_count": 0
            }
        }
        
        # Data lock for thread safety
        self.data_lock = threading.Lock()
        
        # Thumbnail directory
        self.thumbnail_dir = Path("data/cats/thumbnails")
        self.thumbnail_dir.mkdir(exist_ok=True, parents=True)
        
        # Register routes
        self._register_routes()
        
        # Server thread
        self.server_thread = None
        self.running = False
    
    def _create_default_templates(self):
        """Create default templates if they don't exist"""
        # Create index.html template
        index_path = Path(self.app.template_folder) / "index.html"
        if not index_path.exists():
            with open(index_path, 'w') as f:
                f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cat Tracking System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>Cat Tracking System</h1>
            <div class="system-status">
                <span id="status-indicator" class="status-indicator"></span>
                <span id="status-text">Initializing...</span>
            </div>
        </header>
        
        <div class="main-content">
            <div class="detected-cats">
                <h2>Detected Cats</h2>
                <div id="cat-cards" class="cat-cards">
                    <!-- Cat cards will be dynamically inserted here -->
                </div>
            </div>
            
            <div class="statistics">
                <h2>Statistics</h2>
                <div class="stat-container">
                    <div class="stat-card">
                        <h3>System Uptime</h3>
                        <p id="uptime">0s</p>
                    </div>
                    <div class="stat-card">
                        <h3>Total Detections</h3>
                        <p id="detection-count">0</p>
                    </div>
                    <div class="stat-card">
                        <h3>Total Cats</h3>
                        <p id="cat-count">0</p>
                    </div>
                </div>
                
                <div class="chart-container">
                    <canvas id="time-chart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='dashboard.js') }}"></script>
</body>
</html>
                """)
        
        # Create CSS file
        css_path = Path(self.app.static_folder) / "style.css"
        if not css_path.exists():
            with open(css_path, 'w') as f:
                f.write("""
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f4f4f4;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ddd;
}

h1, h2, h3 {
    color: #333;
}

.system-status {
    display: flex;
    align-items: center;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #ccc;
    margin-right: 8px;
}

.status-indicator.active {
    background-color: #4CAF50;
}

.status-indicator.inactive {
    background-color: #f44336;
}

.main-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

.detected-cats, .statistics {
    background: white;
    padding: 20px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.cat-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.cat-card {
    border: 1px solid #ddd;
    border-radius: 5px;
    overflow: hidden;
    transition: transform 0.2s;
}

.cat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.cat-image {
    width: 100%;
    height: 150px;
    object-fit: cover;
    background-color: #eee;
}

.cat-details {
    padding: 10px;
}

.cat-name {
    font-weight: bold;
    margin-bottom: 5px;
}

.cat-info {
    font-size: 0.9em;
    color: #666;
}

.stat-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-bottom: 20px;
}

.stat-card {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 5px;
    text-align: center;
}

.stat-card h3 {
    font-size: 1em;
    margin-bottom: 5px;
}

.stat-card p {
    font-size: 1.5em;
    font-weight: bold;
    color: #2196F3;
}

.chart-container {
    margin-top: 20px;
    height: 250px;
}

@media (max-width: 768px) {
    .main-content {
        grid-template-columns: 1fr;
    }
    
    .stat-container {
        grid-template-columns: 1fr;
    }
}
                """)
    
    def _register_routes(self):
        """Register Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('index.html')
        
        @self.app.route('/api/data')
        def get_data():
            """API endpoint to get current data"""
            with self.data_lock:
                # Calculate uptime
                current_time = time.time()
                self.data["system_status"]["uptime"] = current_time - self.data["system_status"]["start_time"]
                return jsonify(self.data)
        
        @self.app.route('/api/thumbnail/<cat_id>')
        def get_thumbnail(cat_id):
            """API endpoint to get cat thumbnail"""
            thumbnail_path = self.thumbnail_dir / f"{cat_id}.jpg"
            
            if thumbnail_path.exists():
                return send_from_directory(self.thumbnail_dir, f"{cat_id}.jpg")
            else:
                # Return a placeholder image
                return send_from_directory(self.app.static_folder, "cat_placeholder.jpg")
        
        @self.app.route('/api/cats/<cat_id>')
        def get_cat(cat_id):
            """API endpoint to get specific cat data"""
            with self.data_lock:
                if cat_id in self.data["cats"]:
                    return jsonify(self.data["cats"][cat_id])
                else:
                    return jsonify({"error": "Cat not found"}), 404
        
        @self.app.route('/api/rename_cat', methods=['POST'])
        def rename_cat():
            """API endpoint to rename a cat"""
            data = request.json
            cat_id = data.get('cat_id')
            new_name = data.get('name')
            
            if not cat_id or not new_name:
                return jsonify({"error": "Missing cat_id or name"}), 400
            
            with self.data_lock:
                if cat_id in self.data["cats"]:
                    self.data["cats"][cat_id]["name"] = new_name
                    return jsonify({"success": True})
                else:
                    return jsonify({"error": "Cat not found"}), 404
    
    def start(self):
        """Start the web dashboard server"""
        if self.running:
            self.logger.warning("Web dashboard already running")
            return
        
        # Update system status
        with self.data_lock:
            self.data["system_status"]["running"] = True
        
        # Create placeholder image if not exists
        placeholder_path = Path(self.app.static_folder) / "cat_placeholder.jpg"
        if not placeholder_path.exists():
            self._create_placeholder_image(placeholder_path)
        
        # Start server in a separate thread
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.logger.info(f"Web dashboard started on http://{self.host}:{self.port}")
    
    def _run_server(self):
        """Run the Flask server"""
        try:
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        except Exception as e:
            self.logger.error(f"Error running web server: {str(e)}")
            self.running = False
    
    def stop(self):
        """Stop the web dashboard server"""
        if not self.running:
            return
        
        # Update system status
        with self.data_lock:
            self.data["system_status"]["running"] = False
        
        # Stop server (in a production setting, we would use a proper shutdown mechanism)
        self.running = False
        if self.server_thread:
            # Flask doesn't have a clean way to stop from another thread
            # In a production environment, use a proper WSGI server like gunicorn
            # This is a simple implementation for the prototype
            self.logger.info("Stopping web dashboard (might take a few seconds)...")
        
        self.logger.info("Web dashboard stopped")
    
    def update_data(self, identified_cats: Dict):
        """
        Update dashboard data with new cat detections
        
        Args:
            identified_cats: Dictionary of identified cats from the detector
        """
        if not self.running:
            return
        
        with self.data_lock:
            # Update detection count
            self.data["system_status"]["detection_count"] += len(identified_cats)
            
            # Update frame count
            self.data["system_status"]["frame_count"] += 1
            
            # Update cat data
            for track_id, cat_data in identified_cats.items():
                cat_id = cat_data["cat_id"]
                
                # Create or update cat entry
                if cat_id not in self.data["cats"]:
                    self.data["cats"][cat_id] = {
                        "name": cat_data["name"],
                        "appearance_count": 1,
                        "total_time_visible": 0,
                        "last_seen": time.time(),
                        "thumbnail": cat_data.get("thumbnail"),
                        "currently_detected": True
                    }
                else:
                    # Update existing cat
                    self.data["cats"][cat_id]["appearance_count"] += 1
                    self.data["cats"][cat_id]["last_seen"] = time.time()
                    self.data["cats"][cat_id]["currently_detected"] = True
                
                # Update time visible
                if "total_time_visible" in cat_data:
                    self.data["cats"][cat_id]["total_time_visible"] = cat_data["total_time_visible"]
                
                # Update thumbnail if available
                if "thumbnail" in cat_data and cat_data["thumbnail"]:
                    self.data["cats"][cat_id]["thumbnail"] = cat_data["thumbnail"]
            
            # Mark cats not currently detected
            current_cat_ids = [cat_data["cat_id"] for cat_data in identified_cats.values()]
            for cat_id in self.data["cats"]:
                if cat_id not in current_cat_ids:
                    self.data["cats"][cat_id]["currently_detected"] = False
            
            # Add entry to history (keep limited history)
            history_entry = {
                "timestamp": time.time(),
                "cat_count": len(set(current_cat_ids)),
                "cat_ids": current_cat_ids
            }
            
            self.data["history"].append(history_entry)
            
            # Keep only last 100 history entries
            if len(self.data["history"]) > 100:
                self.data["history"] = self.data["history"][-100:]
    
    def _create_placeholder_image(self, path: Path):
        """Create a placeholder image for cats without thumbnails"""
        try:
            # Create a simple placeholder with cat silhouette
            img = np.ones((200, 200, 3), dtype=np.uint8) * 240  # Light gray
            
            # Add a simple cat shape (just a circle for simplicity)
            cv2.circle(img, (100, 100), 60, (200, 200, 200), -1)  # Head
            cv2.circle(img, (80, 80), 10, (150, 150, 150), -1)    # Left eye
            cv2.circle(img, (120, 80), 10, (150, 150, 150), -1)   # Right eye
            
            # Add text
            cv2.putText(img, "Cat", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            
            # Save image
            cv2.imwrite(str(path), img)
            self.logger.debug(f"Created placeholder image at {path}")
        except Exception as e:
            self.logger.error(f"Error creating placeholder image: {str(e)}")