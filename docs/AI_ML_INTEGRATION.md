# SMPStats AI & Machine Learning Integration Guide

> **Advanced Data Processing on Top of the SMPStats API**

This guide describes how to build AI and machine learning applications using data from the SMPStats API. The rich event data—coordinates, timestamps, player actions, and spatial patterns—makes SMPStats an ideal data source for advanced analytics.

---

## Overview

SMPStats provides a comprehensive REST API that exposes:

- **Heatmaps**: Spatial activity data (mining, deaths, movement, damage)
- **Timeline Data**: Historical player statistics with daily snapshots
- **Moments**: Significant gameplay events with timestamps and coordinates
- **Social Data**: Player proximity and interaction patterns
- **Health Metrics**: Server performance data

This data can power four categories of AI/ML applications:

| Category | Use Case | Difficulty | Key Data Sources |
|----------|----------|------------|------------------|
| **3D Classification** | Identify built structures | Advanced | Heatmaps, Block logs |
| **Time-Series Prediction** | Predict player actions | Intermediate | Timeline, Moments |
| **Natural Language Generation** | Auto-generate wiki articles | Intermediate | Moments, Social data |
| **Spatial Analysis** | Optimize infrastructure | Beginner | Movement heatmaps |

---

## Prerequisites

### Python Environment

```bash
# Create virtual environment
python -m venv smpstats-ai
source smpstats-ai/bin/activate  # Linux/Mac
# or: smpstats-ai\Scripts\activate  # Windows

# Install core dependencies
pip install requests pandas numpy matplotlib

# For machine learning (choose based on use case)
pip install scikit-learn  # Clustering, basic ML
pip install torch         # Neural networks (PyTorch)
pip install tensorflow    # Neural networks (TensorFlow)
pip install transformers  # LLM integration
```

### API Configuration

```python
import requests

API_BASE = "http://localhost:8765"
API_KEY = "your-api-key-here"

headers = {"X-API-Key": API_KEY}

def api_get(endpoint, params=None):
    """Helper function for API requests."""
    response = requests.get(f"{API_BASE}{endpoint}", headers=headers, params=params)
    response.raise_for_status()
    return response.json()
```

---

## 1. Structure Classification (3D Neural Network)

### Concept

Train a neural network to recognize what players have built (houses, farms, redstone machines) using spatial block data. Minecraft builds are essentially 3D voxel data—perfect for 3D convolutional neural networks.

### Data Pipeline

```
Heatmap Data → Voxelization → 3D Numpy Array → 3D CNN → Classification
```

### Step 1: Fetch Spatial Data

```python
import numpy as np
import pandas as pd

def fetch_mining_heatmap(world="world", grid_size=16):
    """Fetch mining heatmap data from API."""
    data = api_get("/heatmap/MINING", {
        "world": world,
        "grid": grid_size,
        "from": "30d"  # Last 30 days
    })
    return pd.DataFrame(data)

def fetch_position_heatmap(world="world", grid_size=8):
    """Fetch position/movement heatmap for finer detail."""
    data = api_get("/heatmap/POSITION", {
        "world": world,
        "grid": grid_size,
        "from": "7d"
    })
    return pd.DataFrame(data)
```

### Step 2: Voxelization

Convert coordinate data into a 3D numpy array suitable for neural networks:

```python
def voxelize_region(heatmap_df, center_x, center_z, size=64):
    """
    Convert heatmap data to a 3D voxel grid.
    
    Args:
        heatmap_df: DataFrame with x, z, count columns
        center_x, center_z: Center of the region to voxelize
        size: Size of the region in blocks (e.g., 64x64)
    
    Returns:
        3D numpy array (size x 256 x size) with activity counts
    """
    # Initialize empty voxel grid
    voxel = np.zeros((size, 256, size), dtype=np.float32)
    
    half = size // 2
    for _, row in heatmap_df.iterrows():
        # Convert world coordinates to voxel indices
        vx = int(row['x']) - center_x + half
        vz = int(row['z']) - center_z + half
        
        if 0 <= vx < size and 0 <= vz < size:
            # Spread count across Y levels (simplified)
            # In production, you'd want actual Y-coordinate data
            voxel[vx, 64:128, vz] = row['count']
    
    return voxel

# Example usage
heatmap = fetch_mining_heatmap()
voxel_data = voxelize_region(heatmap, center_x=0, center_z=0, size=64)
print(f"Voxel shape: {voxel_data.shape}")  # (64, 256, 64)
```

### Step 3: 3D CNN Model (PyTorch)

```python
import torch
import torch.nn as nn

class StructureClassifier(nn.Module):
    """3D CNN for Minecraft structure classification."""
    
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Input: (1, 64, 256, 64)
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 8, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# Class labels
STRUCTURE_CLASSES = [
    "house",           # 0
    "farm",            # 1
    "mob_grinder",     # 2
    "redstone_machine", # 3
    "storage_room"     # 4
]
```

### Step 4: Training

```python
def train_classifier(model, train_loader, epochs=50):
    """Train the structure classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for voxels, labels in train_loader:
            voxels = voxels.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(voxels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model
```

### Step 5: Inference & Integration

```python
def classify_area(model, x, z, heatmap_df):
    """Classify what type of structure is at the given coordinates."""
    model.eval()
    
    # Voxelize the area
    voxel = voxelize_region(heatmap_df, x, z)
    voxel_tensor = torch.from_numpy(voxel).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(voxel_tensor)
        probabilities = torch.softmax(output, dim=1)
        class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[0, class_idx].item()
    
    return STRUCTURE_CLASSES[class_idx], confidence

# Example: Notify Discord when someone builds something
def on_build_detected(player, x, z, structure_type, confidence):
    """Send notification about detected build."""
    if confidence > 0.85:
        message = f"🏠 {player} is building a {structure_type} at ({x}, {z}) — {confidence:.0%} confident"
        # send_to_discord(message)
        print(message)
```

### Use Cases

- **Admin Moderation**: Detect inappropriate builds automatically
- **Server Statistics**: Track what types of structures players build most
- **Community Highlights**: Automatically feature impressive builds

---

## 2. Next-Action Prediction (LSTM / Time-Series)

### Concept

Predict what a player will do in the next 10 minutes based on their behavior patterns from the last 30 minutes. Uses sequence modeling (LSTM or Transformer) to learn player behavior patterns.

### Data Pipeline

```
Timeline/Moments → Tokenize Actions → Sequences → LSTM → Prediction
```

### Step 1: Fetch Player Timeline

```python
def fetch_player_activity(uuid, days=7):
    """Fetch a player's recent activity timeline."""
    timeline = api_get(f"/timeline/{uuid}", {"limit": days})
    moments = api_get("/moments/query", {
        "player": uuid,
        "from": f"{days}d",
        "limit": 500
    })
    return timeline, moments

def fetch_recent_moments(hours=24):
    """Fetch all recent moments for behavior analysis."""
    return api_get("/moments/recent", {
        "from": f"{hours}h",
        "limit": 1000
    })
```

### Step 2: Tokenize Actions

```python
# Action vocabulary
ACTION_TOKENS = {
    "LOGIN": 1,
    "LOGOUT": 2,
    "MINING": 3,
    "BUILDING": 4,
    "CRAFTING": 5,
    "COMBAT_PVP": 6,
    "COMBAT_MOB": 7,
    "DEATH": 8,
    "MOVEMENT_OVERWORLD": 9,
    "MOVEMENT_NETHER": 10,
    "MOVEMENT_END": 11,
    "AFK": 12,
    "EXPLORATION": 13,
    "FARMING": 14,
    "PAD": 0  # Padding token
}

def tokenize_moments(moments):
    """Convert moments to action token sequence."""
    tokens = []
    for moment in moments:
        moment_type = moment.get("type", "").upper()
        
        # Map moment types to action tokens
        if "DIAMOND" in moment_type or "ORE" in moment_type:
            tokens.append(ACTION_TOKENS["MINING"])
        elif "DEATH" in moment_type:
            tokens.append(ACTION_TOKENS["DEATH"])
        elif "KILL" in moment_type:
            if "PLAYER" in moment_type:
                tokens.append(ACTION_TOKENS["COMBAT_PVP"])
            else:
                tokens.append(ACTION_TOKENS["COMBAT_MOB"])
        elif "BUILD" in moment_type or "PLACE" in moment_type:
            tokens.append(ACTION_TOKENS["BUILDING"])
        elif "CRAFT" in moment_type:
            tokens.append(ACTION_TOKENS["CRAFTING"])
        else:
            tokens.append(ACTION_TOKENS["EXPLORATION"])
    
    return tokens
```

### Step 3: Sequence Preparation

```python
# Constants for model configuration
SEQ_LENGTH = 30  # Number of actions to use as input
PREDICTION_LENGTH = 10  # Number of actions to predict

def create_sequences(tokens, seq_length=SEQ_LENGTH, prediction_length=PREDICTION_LENGTH):
    """
    Create training sequences for LSTM.
    
    Input: 30 actions → Predict: Next 10 actions
    """
    sequences = []
    targets = []
    
    for i in range(len(tokens) - seq_length - prediction_length):
        seq = tokens[i:i + seq_length]
        target = tokens[i + seq_length:i + seq_length + prediction_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)
```

### Step 4: LSTM Model (PyTorch)

```python
class ActionPredictor(nn.Module):
    """LSTM model for next-action prediction."""
    
    def __init__(self, vocab_size=15, embed_dim=32, hidden_dim=64, output_steps=10):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size * output_steps)
        self.output_steps = output_steps
        self.vocab_size = vocab_size
    
    def forward(self, x):
        # x shape: (batch, seq_length)
        embedded = self.embedding(x)  # (batch, seq, embed_dim)
        lstm_out, _ = self.lstm(embedded)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Predict next N actions
        output = self.fc(last_hidden)
        return output.view(-1, self.output_steps, self.vocab_size)
```

### Step 5: Prediction & Alerts

```python
# Constants are defined in Step 3 above: SEQ_LENGTH = 30, PREDICTION_LENGTH = 10

def predict_next_actions(model, recent_actions, num_predictions=PREDICTION_LENGTH):
    """Predict what the player will do next."""
    model.eval()
    
    # Tokenize and prepare input
    tokens = tokenize_moments(recent_actions)
    if len(tokens) < SEQ_LENGTH:
        tokens = [0] * (SEQ_LENGTH - len(tokens)) + tokens  # Pad
    
    input_tensor = torch.tensor(tokens[-SEQ_LENGTH:]).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        predictions = torch.argmax(output, dim=2).squeeze().tolist()
    
    # Convert back to action names
    token_to_action = {v: k for k, v in ACTION_TOKENS.items()}
    return [token_to_action.get(t, "UNKNOWN") for t in predictions]

def smart_assistant_alert(player_uuid, predictions):
    """Send smart alerts based on predictions."""
    if "DEATH" in predictions[:3]:  # Death predicted in next 3 actions
        print(f"⚠️ {player_uuid} might be in danger! Pattern suggests imminent death.")
    
    if predictions.count("MINING") >= 5:
        print(f"⛏️ {player_uuid} is likely starting a mining session.")
    
    if "COMBAT_PVP" in predictions[:5]:
        print(f"⚔️ {player_uuid} may be heading into PvP combat.")
```

### Use Cases

- **Smart Assistant Bot**: Warn players of predicted danger
- **Resource Preloading**: Prepare chunks the player is likely to visit
- **Admin Monitoring**: Detect suspicious behavior patterns

---

## 3. Automatic Server-Wiki Generation (RAG + LLM)

### Concept

Automatically generate wiki articles about server events, locations, and player achievements using Retrieval-Augmented Generation (RAG) with a Large Language Model.

### Data Pipeline

```
Moments + Social Data → Event Clustering → Context Building → LLM → Wiki Article
```

### Step 1: Gather Event Data

```python
def gather_event_context(time_range="7d"):
    """Gather all relevant data for wiki generation."""
    
    # Fetch moments
    moments = api_get("/moments/recent", {"from": time_range, "limit": 500})
    
    # Fetch social pairs
    social = api_get("/social/top", {"limit": 100})
    
    # Fetch player stats
    stats = api_get("/stats/all")
    
    # Fetch death replays for dramatic events
    deaths = api_get("/death/replay", {"limit": 50})
    
    return {
        "moments": moments,
        "social": social,
        "stats": stats,
        "deaths": deaths
    }
```

### Step 2: Event Clustering

```python
from sklearn.cluster import DBSCAN
from datetime import datetime

def cluster_events(moments, eps_minutes=30, min_samples=3):
    """
    Cluster nearby events in time and space.
    Events within 30 minutes and 100 blocks are grouped together.
    """
    if not moments:
        return []
    
    # Prepare features: timestamp, x, z
    features = []
    for m in moments:
        t = m.get("startedAt", 0) / 60000  # Convert to minutes
        x = m.get("x", 0) / 100  # Normalize coordinates
        z = m.get("z", 0) / 100
        features.append([t, x, z])
    
    features = np.array(features)
    
    # Cluster with DBSCAN
    clustering = DBSCAN(eps=eps_minutes, min_samples=min_samples).fit(features)
    
    # Group moments by cluster
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label == -1:  # Noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(moments[i])
    
    return list(clusters.values())
```

### Step 3: Build Context for LLM

```python
def build_event_context(cluster, social_data, stats):
    """Build a structured context object for the LLM."""
    
    # Extract key information
    players = set()
    deaths = 0
    kills = 0
    avg_x, avg_z = 0, 0
    
    for moment in cluster:
        players.add(moment.get("playerId"))
        if "DEATH" in moment.get("type", "").upper():
            deaths += 1
        if "KILL" in moment.get("type", "").upper():
            kills += 1
        avg_x += moment.get("x", 0)
        avg_z += moment.get("z", 0)
    
    n = len(cluster)
    avg_x /= n
    avg_z /= n
    
    # Find social connections
    connections = []
    for pair in social_data:
        if pair["a"] in players or pair["b"] in players:
            connections.append(f"{pair['name_a']} & {pair['name_b']}")
    
    # Build context
    context = {
        "event_count": n,
        "participants": list(players),
        "participant_count": len(players),
        "deaths": deaths,
        "kills": kills,
        "location": {"x": int(avg_x), "z": int(avg_z)},
        "start_time": min(m.get("startedAt", 0) for m in cluster),
        "end_time": max(m.get("endedAt", 0) for m in cluster),
        "social_connections": connections[:5],
        "moment_types": list(set(m.get("type", "") for m in cluster)),
        "raw_moments": cluster[:10]  # First 10 for detail
    }
    
    return context
```

### Step 4: Generate Wiki Article

```python
from transformers import pipeline

class WikiGenerator:
    """Wiki article generator with cached model pipeline."""
    
    def __init__(self, model_name="gpt2"):
        """Initialize the generator once and reuse for all articles."""
        self.generator = pipeline("text-generation", model=model_name, max_length=500)
    
    def generate(self, context):
        """Generate a wiki article using the cached LLM pipeline."""
        prompt = f"""You are a server historian writing a wiki article about a Minecraft server event.

Event Data:
- Participants: {', '.join(context['participants'][:5])}
- Location: X={context['location']['x']}, Z={context['location']['z']}
- Deaths: {context['deaths']}
- Kills: {context['kills']}
- Event Types: {', '.join(context['moment_types'])}
- Social Connections: {', '.join(context['social_connections'])}

Write an engaging wiki article about this event. Include:
1. A dramatic title
2. List of participants with their roles
3. A narrative description of what happened
4. The outcome and significance

Article:"""
        
        result = self.generator(prompt)
        if not result or not result[0].get("generated_text"):
            return "Error: Failed to generate article"
        return result[0]["generated_text"]

# Usage: Initialize once, use many times
# wiki_gen = WikiGenerator("gpt2")
# article1 = wiki_gen.generate(context1)
# article2 = wiki_gen.generate(context2)

# For OpenAI API
def generate_with_openai(context, api_key):
    """Generate using OpenAI API (modern client library)."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    system_prompt = "You are a historian for a Minecraft server, writing engaging wiki articles."
    user_prompt = f"""Write a wiki article about this event:

Participants: {context['participants']}
Location: ({context['location']['x']}, {context['location']['z']})
Deaths: {context['deaths']}, Kills: {context['kills']}
Event Types: {context['moment_types']}

Write in an epic, narrative style. Include a title, participant list, and description."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content
```

### Step 5: Output as HTML

```python
import markdown

def wiki_to_html(article_text, output_path):
    """Convert wiki article to HTML page."""
    
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Server Wiki</title>
    <style>
        body { font-family: Georgia, serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
        .participants { background: #ecf0f1; padding: 10px; border-radius: 5px; }
        .narrative { line-height: 1.8; }
    </style>
</head>
<body>
{content}
</body>
</html>"""
    
    html_content = markdown.markdown(article_text)
    full_html = html_template.format(content=html_content)
    
    with open(output_path, 'w') as f:
        f.write(full_html)
    
    print(f"Wiki article saved to {output_path}")
```

### Use Cases

- **Automatic Server History**: Document major battles, achievements, explorations
- **Weekly Digests**: Generate summaries of server activity
- **Player Profiles**: Auto-generate player biography pages

---

## 4. Pathfinding Optimization & Desire Paths

### Concept

Analyze player movement patterns to discover "desire paths" (natural walking routes) and optimize server infrastructure placement. This is classic data science applied to spatial data.

### Data Pipeline

```
Movement Heatmap → Graph Construction → Path Analysis → Recommendations
```

### Step 1: Fetch Movement Data

```python
def fetch_movement_heatmap(world="world", grid=8, days=30):
    """Fetch fine-grained movement data."""
    data = api_get("/heatmap/POSITION", {
        "world": world,
        "grid": grid,
        "from": f"{days}d"
    })
    return pd.DataFrame(data)
```

### Step 2: Build Weighted Graph

```python
import networkx as nx

def build_movement_graph(heatmap_df, threshold=10):
    """
    Build a graph where nodes are grid cells and edges connect adjacent cells.
    Edge weights are inversely proportional to movement frequency.
    """
    G = nx.Graph()
    
    # Create nodes
    for _, row in heatmap_df.iterrows():
        x, z = int(row['x']), int(row['z'])
        count = row['count']
        G.add_node((x, z), weight=count)
    
    # Create edges between adjacent cells
    for node in G.nodes():
        x, z = node
        neighbors = [
            (x+1, z), (x-1, z), (x, z+1), (x, z-1),
            (x+1, z+1), (x-1, z-1), (x+1, z-1), (x-1, z+1)
        ]
        
        for neighbor in neighbors:
            if G.has_node(neighbor):
                # Weight = inverse of combined traffic (lower = more used)
                weight_a = G.nodes[node].get('weight', 1)
                weight_b = G.nodes[neighbor].get('weight', 1)
                combined = (weight_a + weight_b) / 2
                
                if combined > threshold:
                    # High traffic = low cost
                    edge_weight = 1000 / combined
                else:
                    # Low traffic = high cost
                    edge_weight = 100
                
                G.add_edge(node, neighbor, weight=edge_weight)
    
    return G
```

### Step 3: Find Desire Paths

```python
def find_desire_paths(G, start, end):
    """
    Find the most-used path between two points.
    This represents the 'desire path' - where players naturally walk.
    """
    try:
        path = nx.shortest_path(G, start, end, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None

def find_popular_routes(G, top_n=10):
    """Find the most heavily trafficked routes on the server."""
    
    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # Sort nodes by centrality (most important intersections)
    important_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    
    return important_nodes[:top_n]
```

### Step 4: Compare with Built Infrastructure

```python
def analyze_path_efficiency(movement_graph, infrastructure_coords):
    """
    Compare actual movement patterns with built paths.
    Identify where players walk vs where paths exist.
    """
    
    # Get high-traffic nodes
    high_traffic = [(n, d['weight']) for n, d in movement_graph.nodes(data=True) 
                    if d.get('weight', 0) > 100]
    
    # Check which high-traffic areas have infrastructure
    covered = []
    uncovered = []
    
    for node, traffic in high_traffic:
        # Check if any infrastructure is nearby (within 2 blocks)
        is_covered = any(
            abs(node[0] - ix) <= 2 and abs(node[1] - iz) <= 2
            for ix, iz in infrastructure_coords
        )
        
        if is_covered:
            covered.append((node, traffic))
        else:
            uncovered.append((node, traffic))
    
    return {
        "covered": covered,
        "uncovered": uncovered,
        "coverage_ratio": len(covered) / len(high_traffic) if high_traffic else 0
    }
```

### Step 5: Visualization

```python
import matplotlib.pyplot as plt

def visualize_desire_paths(heatmap_df, recommendations, output_path="desire_paths.png"):
    """Create a visual map of desire paths and recommendations."""
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot heatmap
    scatter = ax.scatter(
        heatmap_df['x'], 
        heatmap_df['z'], 
        c=heatmap_df['count'],
        cmap='hot',
        alpha=0.6,
        s=10
    )
    
    # Mark recommended infrastructure locations
    for rec in recommendations[:10]:
        x, z = rec[0]
        ax.plot(x, z, 'g^', markersize=15, label='Build here')
        ax.annotate(f"Traffic: {rec[1]:.0f}", (x, z), textcoords="offset points", 
                   xytext=(0,10), ha='center')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_title('Player Movement Heatmap & Recommended Paths')
    plt.colorbar(scatter, label='Traffic Intensity')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

def generate_recommendations(analysis_result, grid_size=8):
    """Generate infrastructure recommendations."""
    
    recommendations = []
    
    # Sort uncovered areas by traffic
    uncovered = sorted(analysis_result['uncovered'], key=lambda x: x[1], reverse=True)
    
    for node, traffic in uncovered[:5]:
        x, z = node
        rec = {
            "location": {"x": x * grid_size, "z": z * grid_size},  # Convert back to world coords
            "traffic": traffic,
            "suggestion": "Build path or bridge here",
            "priority": "high" if traffic > 500 else "medium"
        }
        recommendations.append(rec)
    
    return recommendations
```

### Use Cases

- **Infrastructure Planning**: Build roads where players actually walk
- **Spawn Design**: Optimize spawn layout based on movement patterns
- **Performance Optimization**: Pre-load chunks on popular routes

---

## Complete Example: Weekly Analytics Pipeline

```python
#!/usr/bin/env python3
"""
Weekly SMPStats Analytics Pipeline

Run this script weekly to:
1. Analyze player behavior patterns
2. Detect significant events
3. Generate wiki articles
4. Create infrastructure recommendations
"""

import json
from datetime import datetime

def weekly_analysis():
    """Run complete weekly analytics."""
    
    print(f"=== SMPStats Weekly Analysis ({datetime.now().isoformat()}) ===\n")
    
    # 1. Gather Data
    print("📊 Gathering data from API...")
    context = gather_event_context("7d")
    print(f"   - {len(context['moments'])} moments")
    print(f"   - {len(context['stats'])} players tracked")
    print(f"   - {len(context['deaths'])} death replays")
    
    # 2. Cluster Events
    print("\n🔍 Clustering events...")
    clusters = cluster_events(context['moments'])
    print(f"   - Found {len(clusters)} significant event clusters")
    
    # 3. Generate Wiki Articles
    print("\n📝 Generating wiki articles...")
    wiki_gen = WikiGenerator()  # Initialize once
    for i, cluster in enumerate(clusters[:3]):  # Top 3 events
        event_context = build_event_context(cluster, context['social'], context['stats'])
        article = wiki_gen.generate(event_context)
        wiki_to_html(article, f"wiki/event_{i+1}.html")
    
    # 4. Analyze Movement Patterns
    print("\n🗺️ Analyzing movement patterns...")
    movement = fetch_movement_heatmap()
    graph = build_movement_graph(movement)
    
    popular = find_popular_routes(graph)
    print(f"   - Found {len(popular)} high-traffic intersections")
    
    # 5. Generate Recommendations
    print("\n💡 Generating recommendations...")
    # Load infrastructure coordinates from your server's path/road data
    # Example: Query your Dynmap data or parse a world map file
    infrastructure = []  # Replace with: load_infrastructure_from_dynmap() or similar
    analysis = analyze_path_efficiency(graph, infrastructure)
    
    recommendations = generate_recommendations(analysis)
    for rec in recommendations:
        print(f"   - Build at ({rec['location']['x']}, {rec['location']['z']}) - {rec['priority']} priority")
    
    # 6. Save Report
    report = {
        "generated_at": datetime.now().isoformat(),
        "event_clusters": len(clusters),
        "infrastructure_coverage": analysis['coverage_ratio'],
        "recommendations": recommendations
    }
    
    with open("weekly_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ Weekly analysis complete! Report saved to weekly_report.json")

if __name__ == "__main__":
    weekly_analysis()
```

---

## Tech Stack Summary

| Component | Recommended Tools | Purpose |
|-----------|-------------------|---------|
| **Data Fetching** | `requests`, `pandas` | API consumption, data manipulation |
| **ML Framework** | `PyTorch` or `TensorFlow` | Neural networks (CNN, LSTM) |
| **Clustering** | `scikit-learn` (DBSCAN, K-Means) | Event grouping |
| **Graph Analysis** | `networkx` | Pathfinding, centrality |
| **LLM Integration** | `transformers`, OpenAI API, Ollama | Text generation |
| **Visualization** | `matplotlib`, `plotly` | Charts, heatmaps |

---

## Getting Started Recommendations

| Your Goal | Start With | Difficulty |
|-----------|------------|------------|
| Visual insights | **Desire Paths** (Section 4) | ⭐ Beginner |
| Learn neural networks | **Action Prediction** (Section 2) | ⭐⭐ Intermediate |
| Content automation | **Wiki Generation** (Section 3) | ⭐⭐ Intermediate |
| Advanced AI | **Structure Classification** (Section 1) | ⭐⭐⭐ Advanced |

---

## API Endpoints Reference

For AI/ML applications, these endpoints are most useful:

| Endpoint | Data Type | ML Use Case |
|----------|-----------|-------------|
| `/heatmap/MINING` | Spatial | Structure classification, resource analysis |
| `/heatmap/POSITION` | Spatial | Movement analysis, desire paths |
| `/heatmap/DEATH` | Spatial | Danger zone detection |
| `/moments/recent` | Temporal | Action prediction, event clustering |
| `/timeline/{uuid}` | Temporal | Player behavior modeling |
| `/social/top` | Relational | Group detection, social graphs |
| `/stats/all` | Aggregate | Player profiling, anomaly detection |

See [API.md](API.md) for complete endpoint documentation.

---

## Contributing

Have you built something cool with SMPStats data? We'd love to hear about it!

- **Share your project**: Open a GitHub Discussion
- **Contribute examples**: Submit a PR with your code
- **Report issues**: If the API doesn't provide data you need

---

<p align="center">
  <strong>Turn your Minecraft server into an AI playground.</strong><br>
  <em>The data is there. The possibilities are endless.</em>
</p>
