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




