# Holistic QC rewrite

> Contains a more modular approach to QCing, alleviates the headaches of old stuff


## Requirements
- process batches of txt files by turning them into csv
    - requires basic qcs
    - requires scoring criteria
- plot information on a graph
- save the data in the correct location
- uploads the data to git and server
- adds plots and scoring to github pages


## Plan

- `pull_handler` returns a list of txt files
- `utils` contains commonly used functions like converting txt file to csv
- each domain has its own qc file with diff methods for qcing by task
    - takes in a list of files as an arg and processes them, returning the usability score and logging any problems


## Tasks
- [x] finish cc algos
- [x] test
- [ ] start WL/DWL algos -> separate class from mem



## Relational Database Design Summary for Clinical Trial Cognitive Data

>>Purpose & Scope
	•	This database will organize and store clinical trial cognitive data.
	•	Each participant completes 13 cognitive tasks over two runs each.
	•	The data will be ingested daily from a prewritten backend.
	•	The database will integrate with a frontend using Python and Azure.
	•	Expected data volume: Hundreds to thousands of participants.

>>Core Entities & Relationships

1. Participants (participants)
	•	Stores participant identifiers, their assigned study type (observation/intervention), and their site location.
	•	Each participant completes 26 runs total (13 tasks × 2 runs).
	•	Relationships:
	•	Linked to sites (site_id)
	•	Linked to study_types (study_id)
	•	Has many runs

2. Study Types (study_types)
	•	Defines whether a participant is in the Intervention or Observation group.

3. Sites (sites)
	•	Stores the location each participant is from.
	•	Explicitly defined in the directory structure.

4. Tasks (tasks)
	•	Stores the 13 predefined tasks in a static table.

5. Runs (runs)
	•	Stores each task run per participant (26 runs per participant).
	•	Each run is linked to a participant and a task.
	•	Can store a timestamp (nullable, extracted from CSVs).

6. Results (results)
	•	Stores raw cognitive task data extracted from CSV files.
	•	CSV contents will be stored directly in the database (not just file paths).
	•	Linked to runs via run_id.

7. Reports (reports)
	•	Stores 1-2 PNG files per run as binary blobs (not file paths).
	•	Linked to runs via run_id.
	•	Has a missing_png_flag to track if files are absent.

Constraints & Data Integrity
	•	Primary Keys (PKs) & Foreign Keys (FKs):
	•	participant_id → Primary key in participants
	•	task_id → Primary key in tasks
	•	run_id → Primary key in runs, foreign key links to participants & tasks
	•	result_id → Primary key in results, foreign key links to runs
	•	report_id → Primary key in reports, foreign key links to runs
	•	Data Rules & Validation:
	•	All 13 tasks must be associated with each participant (26 runs total).
	•	missing_png_flag will track missing PNG files.
	•	csv_data will be stored as structured data (likely JSON or table format).

>>Indexing & Optimization

	•	Indexes on:
	•	participant_id (for quick retrieval of participant data)
	•	task_id (for filtering task-based results)
	•	study_id (for intervention vs. observation analysis)
	•	site_id (for location-based analysis)
	•	Storage Considerations:
	•	CSV data stored as structured content (JSON or column format).
	•	PNG files stored as binary blobs.
	•	Query Optimization:
	•	JOINs will be used for participant-level queries.
	•	Materialized views can be considered for frequently used summaries.

>>Security & Access Control
	•	Currently, only you will use the database, so permissions are simple.
	•	Future security measures:
	•	Row-level security for multiple users.
	•	Encryption for sensitive participant records.

>>Backup & Recovery
	•	Daily backups of database storage + binary files.
	•	Azure Blob Storage or PostgreSQL Large Objects for efficient handling of PNG & CSV files.

Next Step: SQL Schema Implementation

Would you like the SQL schema to be written for PostgreSQL, MySQL, or another database system?
