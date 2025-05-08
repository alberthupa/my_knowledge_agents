# Plan to Create Azure Function for Newsletter Ingestion

This document outlines the plan to create an Azure Function that will run daily to fetch new Gmail newsletters since the last download date and upload them to Azure Cosmos DB.

**Objective:** Create an Azure Function triggered daily to ingest new Gmail newsletters into Azure Cosmos DB, starting from the day after the last ingested newsletter or from 7 days ago if no previous data exists.

**Plan Steps:**

1.  **Analyze Existing Code:** Review the provided `src/ingestion_scripts/upload_old_newsletters.py` and the `get_last_newsletter_date` snippet to understand the core logic for fetching, processing, uploading, and determining the last downloaded date.
2.  **Azure Function Structure:** Create a new Python file, `get_newsletters_on_function.py`, which will contain the Azure Function code. This file will include the main function triggered by an Azure Timer Trigger.
3.  **Client Initialization:** Inside the Azure Function's main entry point, initialize the necessary clients: Gmail API service, Azure Cosmos DB client, and the OpenAI Embeddings client. This will involve adapting the initialization logic from `upload_old_newsletters.py` to work within the function context, likely reading environment variables configured in the Azure Function App settings.
4.  **Determine Last Downloaded Date:** Implement or integrate the `get_last_newsletter_date` function to query your Cosmos DB container and retrieve the maximum `chunk_date` for items with `source = 'gmail_newsletter'`.
5.  **Calculate Ingestion Date Range:** Based on the last downloaded date retrieved from Cosmos DB, calculate the date range for which to fetch new newsletters.
    *   If a last downloaded date is found, the start date will be the day *after* the last downloaded date.
    *   If no last downloaded date is found, the start date will be today minus 7 days.
    *   The end date will be the current day.
6.  **Iterate and Process:** Loop through each date within the calculated ingestion date range. For each date, call the core processing logic (adapted from the `process_newsletters_for_date` function in your existing script) to:
    *   Fetch newsletters from Gmail for that specific date.
    *   Clean and chunk the email content.
    *   Generate embeddings for the chunks.
    *   Upload the processed chunks to the Azure Cosmos DB container.
7.  **Error Handling and Logging:** Add robust error handling and logging throughout the function to monitor its execution and diagnose any issues. Use the standard Python `logging` module, which is integrated with Azure Functions monitoring.
8.  **Environment Configuration:** Note that the necessary environment variables (like `COSMOS_CONNECTION_STRING`, Gmail credentials, etc.) will need to be configured in the Azure Function App settings once deployed.
9.  **Save Code:** Save the final Python code for the Azure Function in the specified file path: [`get_newsletters_on_function.py`](get_newsletters_on_function.py).

**Visual Representation of the Plan:**

```mermaid
graph TD
    A[Azure Function Timer Trigger<br>Daily Execution] --> B{Initialize Clients<br>Gmail, Cosmos DB, Embeddings};
    B --> C{Clients Initialized Successfully?};
    C -- No --> D[Log Error and Exit Function];
    C -- Yes --> E[Get Last Downloaded Date<br>from Cosmos DB];
    E --> F{Last Date Found?};
    F -- Yes --> G[Calculate Start Date<br>Last Date + 1 Day];
    F -- No --> H[Calculate Start Date<br>Today - 7 Days];
    G --> I[Set End Date to Today];
    H --> I;
    I --> J{Iterate from Start Date to End Date};
    J --> K[Process Newsletters for Current Date<br>Fetch from Gmail, Clean, Chunk, Embed, Upload to Cosmos DB];
    K --> J;
    J -- Loop Ends --> L[Log Successful Completion];