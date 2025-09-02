# V1 Test Checklist - Critical Items to Verify

## Environment Setup
- [ ] Virtual environment created and activated
- [ ] All required packages installed (crewai, openai, jsonschema, etc.)
- [ ] OpenAI API key properly set in environment variables
- [ ] Python version compatibility verified (3.8+)

## Project Structure
- [ ] Directory structure matches the planned layout
- [ ] All `__init__.py` files created for proper Python packaging
- [ ] Relative imports working correctly between modules

## Fixtures & Data
- [ ] `fixtures/values.json` created with exact data from test cases
- [ ] `fixtures/tasks.v1.json` created with all 50 test tasks
- [ ] Data types match expected outputs exactly
- [ ] No trailing spaces or formatting issues in fixtures

## Tool Implementation
- [ ] All 20 variable tools implemented with correct JSON schemas
- [ ] All 30 function tools implemented with correct JSON schemas
- [ ] JSON Schema validation working (rejects extra properties)
- [ ] Tool outputs match expected types exactly
- [ ] Error handling for invalid inputs implemented

## Core Framework
- [ ] `schemas.py` - JSON Schema definitions working
- [ ] `runner.py` - Orchestrator adapter interface defined
- [ ] `oracle.py` - Exact match validation working
- [ ] `logger.py` - CSV and transcript logging working
- [ ] `run_matrix.py` - Test execution loop working

## CrewAI Adapter
- [ ] `adapters/crewai.py` implemented
- [ ] Tool registration working correctly
- [ ] System prompt properly configured
- [ ] LLM parameters set to deterministic values (temp=0, top_p=0)
- [ ] Tool call execution working
- [ ] Response parsing working

## Test Execution
- [ ] Single task execution working
- [ ] Multiple task execution working
- [ ] Tool usage tracking working
- [ ] Step counting working
- [ ] Timing measurement working
- [ ] Error capture working

## Validation & Logging
- [ ] Expected vs actual output comparison working
- [ ] CSV output generation working
- [ ] Transcript JSONL generation working
- [ ] Error categorization working
- [ ] Tool call logging working

## Data Integrity
- [ ] All 50 test cases can be executed
- [ ] Expected outputs match actual outputs exactly
- [ ] No data corruption between test runs
- [ ] Fixtures remain unchanged during execution

## Performance & Monitoring
- [ ] Memory usage tracking (if possible)
- [ ] Tool call overhead measurement
- [ ] Timeout handling working
- [ ] Step limit enforcement working

## Cross-Platform Preparation
- [ ] Adapter interface generic enough for other platforms
- [ ] Tool catalog easily swappable
- [ ] Fixtures platform-agnostic
- [ ] Logging format standardized

## Pre-Run Verification
- [ ] Run a single simple task (K=1) manually
- [ ] Verify tool outputs match fixtures exactly
- [ ] Check CSV output format
- [ ] Verify transcript logging
- [ ] Test error handling with invalid inputs

## Critical Success Factors
- [ ] **Determinism**: Same inputs always produce same outputs
- [ ] **Validation**: Invalid tool calls are properly rejected
- [ ] **Logging**: Complete audit trail for every test run
- [ ] **Reproducibility**: Tests can be re-run with identical results
- [ ] **Scalability**: Framework can handle multiple orchestrators
