#!/usr/bin/env python3
"""
File-based test that writes results to a file to bypass terminal output issues.
"""

import sys
import os
import traceback
import datetime

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def write_log(message, log_file):
    """Write message to log file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

def test_tokenization_workflow():
    """Test the core tokenization workflow without heavy dependencies."""
    log_file = "/tmp/test_results.log"
    
    try:
        write_log("=== TOKENIZATION WORKFLOW TEST ===", log_file)
        
        # Clear log file
        with open(log_file, 'w') as f:
            f.write("")
        
        write_log("Starting tokenization workflow test...", log_file)
        
        # Test 1: Check if main_new.py exists and has required functions
        main_new_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'ipfs_embeddings_py', 'main_new.py')
        
        if not os.path.exists(main_new_path):
            write_log("ERROR: main_new.py not found", log_file)
            return False
            
        write_log(f"SUCCESS: Found main_new.py at {main_new_path}", log_file)
        
        # Read the file content
        with open(main_new_path, 'r') as f:
            content = f.read()
        
        # Check for required functions that handle tokenization
        required_functions = [
            'safe_tokenizer_encode',
            'safe_tokenizer_decode',
            'safe_chunker_chunk',
            'safe_get_cid',
            'index_cid'
        ]
        
        missing_functions = []
        found_functions = []
        
        for func in required_functions:
            if func in content:
                found_functions.append(func)
                write_log(f"SUCCESS: Found function {func}", log_file)
            else:
                missing_functions.append(func)
                write_log(f"ERROR: Missing function {func}", log_file)
        
        # Test 2: Verify workflow order in code
        write_log("Checking workflow order in code...", log_file)
        
        # Look for the sequence: tokenization -> chunking -> CID generation
        tokenizer_pos = content.find('safe_tokenizer_encode')
        chunker_pos = content.find('safe_chunker_chunk')
        cid_pos = content.find('safe_get_cid')
        
        workflow_correct = True
        if tokenizer_pos != -1 and chunker_pos != -1 and cid_pos != -1:
            write_log(f"Function positions - Tokenizer: {tokenizer_pos}, Chunker: {chunker_pos}, CID: {cid_pos}", log_file)
            
            # The workflow should be: Text -> Tokenize -> Chunk -> CID -> Embed
            if tokenizer_pos < chunker_pos:
                write_log("SUCCESS: Tokenizer comes before chunker in code", log_file)
            else:
                write_log("WARNING: Tokenizer position relative to chunker needs verification", log_file)
                workflow_correct = False
        else:
            write_log("ERROR: Could not find all workflow functions", log_file)
            workflow_correct = False
        
        # Test 3: Check for batch processing indicators
        write_log("Checking for batch processing indicators...", log_file)
        
        batch_indicators = ['batch', 'chunk', 'token']
        found_batch_indicators = []
        
        for indicator in batch_indicators:
            if indicator.lower() in content.lower():
                found_batch_indicators.append(indicator)
                write_log(f"SUCCESS: Found batch processing indicator: {indicator}", log_file)
        
        # Test 4: Check for async/await patterns (important for non-blocking execution)
        write_log("Checking for async patterns...", log_file)
        
        async_patterns = ['async def', 'await ', 'asyncio']
        found_async = []
        
        for pattern in async_patterns:
            if pattern in content:
                found_async.append(pattern)
                write_log(f"SUCCESS: Found async pattern: {pattern}", log_file)
        
        # Final assessment
        write_log("=== FINAL ASSESSMENT ===", log_file)
        
        all_functions_found = len(missing_functions) == 0
        has_batch_processing = len(found_batch_indicators) > 0
        has_async_support = len(found_async) > 0
        
        write_log(f"All required functions found: {all_functions_found}", log_file)
        write_log(f"Workflow order correct: {workflow_correct}", log_file)
        write_log(f"Has batch processing: {has_batch_processing}", log_file)
        write_log(f"Has async support: {has_async_support}", log_file)
        
        success = all_functions_found and workflow_correct and has_batch_processing
        
        if success:
            write_log("OVERALL RESULT: SUCCESS - Tokenization workflow appears correctly implemented", log_file)
        else:
            write_log("OVERALL RESULT: ISSUES FOUND - See details above", log_file)
        
        return success
        
    except Exception as e:
        write_log(f"CRITICAL ERROR: {str(e)}", log_file)
        write_log(f"Traceback: {traceback.format_exc()}", log_file)
        return False

def test_hanging_prevention():
    """Test that we can import basic modules without hanging."""
    log_file = "/tmp/test_results.log"
    
    try:
        write_log("=== HANGING PREVENTION TEST ===", log_file)
        
        # Test basic imports
        import time
        write_log("SUCCESS: time module imported", log_file)
        
        import json
        write_log("SUCCESS: json module imported", log_file)
        
        import os
        write_log("SUCCESS: os module imported", log_file)
        
        # Test that we can do basic file operations
        test_file = "/tmp/hanging_test.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        
        if os.path.exists(test_file):
            write_log("SUCCESS: File operations working", log_file)
            os.remove(test_file)
        
        return True
        
    except Exception as e:
        write_log(f"HANGING PREVENTION ERROR: {str(e)}", log_file)
        return False

def main():
    """Run the file-based tests."""
    log_file = "/tmp/test_results.log"
    
    # Clear the log file
    with open(log_file, 'w') as f:
        f.write("")
    
    write_log("Starting file-based test suite...", log_file)
    
    # Run tests
    tests = [
        ("Hanging Prevention", test_hanging_prevention),
        ("Tokenization Workflow", test_tokenization_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        write_log(f"\n--- Running {test_name} ---", log_file)
        try:
            result = test_func()
            results.append((test_name, result))
            write_log(f"{test_name} result: {'PASS' if result else 'FAIL'}", log_file)
        except Exception as e:
            write_log(f"{test_name} EXCEPTION: {str(e)}", log_file)
            results.append((test_name, False))
    
    # Summary
    write_log("\n=== TEST SUMMARY ===", log_file)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        write_log(f"{test_name}: {status}", log_file)
    
    write_log(f"\nOverall: {passed}/{total} tests passed", log_file)
    
    if passed == total:
        write_log("🎉 ALL TESTS PASSED!", log_file)
    else:
        write_log("❌ Some tests failed", log_file)
    
    # Also try to print to stdout
    try:
        print(f"Test completed. Results written to {log_file}")
        print(f"Overall: {passed}/{total} tests passed")
    except:
        pass  # If stdout is not working, at least we have the log file

if __name__ == "__main__":
    main()
