-- Schema for nanodex knowledge graph
-- Stores code symbols as nodes and their relationships as edges

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT,
    lang TEXT,
    properties TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    properties TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_id, target_id, relationship),
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_path ON nodes(path);
CREATE INDEX IF NOT EXISTS idx_nodes_lang ON nodes(lang);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_source_rel ON edges(source_id, relationship);
CREATE INDEX IF NOT EXISTS idx_edges_target_rel ON edges(target_id, relationship);
CREATE INDEX IF NOT EXISTS idx_edges_relationship ON edges(relationship);

-- View for easy graph inspection
CREATE VIEW IF NOT EXISTS graph_stats AS
SELECT
    (SELECT COUNT(*) FROM nodes) AS total_nodes,
    (SELECT COUNT(*) FROM edges) AS total_edges,
    (SELECT COUNT(DISTINCT type) FROM nodes) AS unique_node_types,
    (SELECT COUNT(DISTINCT relationship) FROM edges) AS unique_edge_types;

-- View for node type distribution
CREATE VIEW IF NOT EXISTS node_type_distribution AS
SELECT
    type,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM nodes), 2) AS percentage
FROM nodes
GROUP BY type
ORDER BY count DESC;

-- View for edge relationship distribution
CREATE VIEW IF NOT EXISTS edge_relationship_distribution AS
SELECT
    relationship,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM edges), 2) AS percentage
FROM edges
GROUP BY relationship
ORDER BY count DESC;
