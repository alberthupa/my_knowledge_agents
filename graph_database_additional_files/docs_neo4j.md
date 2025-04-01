# Neo4j Docker Setup Notes

This document outlines the steps to set up Neo4j using Docker Compose, including directory preparation, container startup, and plugin verification.

## 1. Prepare Directories and Permissions

Before starting the container, create the necessary directories for Neo4j data, logs, import files, and plugins. Also, ensure the correct permissions are set.

```bash
# Create directories
mkdir -p graph_database/{data,logs,import,plugins}

# Set ownership (replace $USER with your actual username if needed)
sudo chown -R $USER:$USER graph_database/

# Set permissions
chmod -R 755 graph_database/
```

*These commands assume you have a `docker-compose.yml` file configured for your Neo4j setup.*

## 2. Start the Neo4j Container

With the directories prepared and permissions set, start the Neo4j container using Docker Compose in detached mode:

```bash
docker-compose up -d
```

## 3. Access Neo4j Browser

Once the container is running:

-   Open your web browser and navigate to `http://localhost:7474`.
-   Log in using the username `neo4j` and the password specified in your `docker-compose.yml` file.

## 4. Verify Plugin Installation

To confirm that the APOC and Graph Data Science (GDS) plugins are installed and working correctly, execute the following Cypher queries in the Neo4j Browser:

### Verify APOC Plugin

```cypher
RETURN apoc.version()
```

This command should return the installed version of the APOC library.

### Verify GDS Plugin

```cypher
RETURN gds.version()
```

This command should return the installed version of the Graph Data Science library.

## References

-   [Deployment of Neo4j Docker Container with APOC and Graph Algorithms Plugins](https://medium.com/swlh/deployment-of-neo4j-docker-container-with-apoc-and-graph-algorithms-plugins-bf48226928f4?utm_source=chatgpt.com)
-   [Simple Graph Database Setup with Neo4j and Docker Compose](https://medium.com/%40matthewghannoum/simple-graph-database-setup-with-neo4j-and-docker-compose-061253593b5a?utm_source=chatgpt.com)
-   [Neo4j Docs: GDS Installation with Docker](https://neo4j.com/docs/graph-data-science/current/installation/installation-docker/?utm_source=chatgpt.com)
