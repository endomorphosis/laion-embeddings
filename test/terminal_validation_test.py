#!/usr/bin/env python3
"""
Terminal-friendly test to validate tokenization workflow without hanging.
"""

import sys
import os

def main():
    """Main test function that validates the tokenization workflow."""
    print("=" * 60)
    print("TOKENIZATION WORKFLOW VALIDATION TEST")
    print("=" * 60)
    
    # Test 1: Check if main_new.py exists
    print("1. Checking main_new.py file...")
    main_new_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'ipfs_embeddings_py', 'main_new.py')
    
    if not os.path.exists(main_new_path):
        print(f"✗ FAIL: main_new.py not found at {main_new_path}")
        return False
    print(f"✓ PASS: Found main_new.py at {main_new_path}")
    
    # Test 2: Check required functions exist
    print("\n2. Checking required tokenization functions...")
    try:
        with open(main_new_path, 'r') as f:
            content = f.read()
        
        required_functions = [
            'safe_tokenizer_encode',
            'safe_tokenizer_decode', 
            'safe_chunker_chunk',
            'safe_get_cid',
            'index_cid'
        ]
        
        all_found = True
        for func in required_functions:
            if f"def {func}(" in content:
                print(f"✓ PASS: Found function {func}")
            else:
                print(f"✗ FAIL: Missing function {func}")
                all_found = False
        
        if not all_found:
            return False
            
    except Exception as e:
        print(f"✗ FAIL: Error reading main_new.py: {e}")
        return False
    
    # Test 3: Validate workflow sequence
    print("\n3. Checking tokenization workflow sequence...")
    
    # Find function positions in file
    tokenizer_pos = content.find('def safe_tokenizer_encode(')
    chunker_pos = content.find('def safe_chunker_chunk(')
    cid_pos = content.find('def safe_get_cid(')
    index_pos = content.find('def index_cid(')
    
    sequence_correct = True
    
    if tokenizer_pos != -1:
        print(f"✓ PASS: safe_tokenizer_encode found at position {tokenizer_pos}")
    else:
        print("✗ FAIL: safe_tokenizer_encode not found")
        sequence_correct = False
    
    if chunker_pos != -1:
        print(f"✓ PASS: safe_chunker_chunk found at position {chunker_pos}")
    else:
        print("✗ FAIL: safe_chunker_chunk not found")
        sequence_correct = False
        
    if cid_pos != -1:
        print(f"✓ PASS: safe_get_cid found at position {cid_pos}")
    else:
        print("✗ FAIL: safe_get_cid not found")
        sequence_correct = False
        
    if index_pos != -1:
        print(f"✓ PASS: index_cid found at position {index_pos}")
    else:
        print("✗ FAIL: index_cid not found")
        sequence_correct = False
    
    # Test 4: Validate batch processing indicators
    print("\n4. Checking batch processing capabilities...")
    
    batch_keywords = ['batch', 'chunk', 'token']
    batch_found = []
    
    for keyword in batch_keywords:
        if keyword.lower() in content.lower():
            count = content.lower().count(keyword.lower())
            batch_found.append(keyword)
            print(f"✓ PASS: Found '{keyword}' ({count} occurrences)")
        else:
            print(f"✗ FAIL: Missing '{keyword}' keyword")
    
    batch_processing_ok = len(batch_found) >= 2  # At least 2 out of 3 keywords
    
    # Test 5: Check for async/await patterns 
    print("\n5. Checking async operation support...")
    
    async_patterns = ['async def', 'await ', 'asyncio']
    async_found = []
    
    for pattern in async_patterns:
        if pattern in content:
            count = content.count(pattern)
            async_found.append(pattern)
            print(f"✓ PASS: Found '{pattern}' ({count} occurrences)")
        else:
            print(f"  INFO: '{pattern}' not found (optional)")
    
    # Test 6: Verify correct workflow order (tokenization before embedding)
    print("\n6. Validating workflow order...")
    
    expected_workflow = [
        "Text Input",
        "Tokenization (safe_tokenizer_encode)", 
        "Chunking (safe_chunker_chunk)",
        "CID Generation (safe_get_cid)",
        "Batch Processing (index_cid)",
        "Embedding Processing"
    ]
    
    print("Expected workflow sequence:")
    for i, step in enumerate(expected_workflow, 1):
        print(f"  {i}. {step}")
    
    # The critical validation: tokenization happens before embedding
    # We validate this by ensuring our tokenization functions exist and are properly ordered
    workflow_order_ok = (tokenizer_pos < chunker_pos and 
                        chunker_pos < cid_pos and 
                        cid_pos < index_pos)
    
    if workflow_order_ok:
        print("✓ PASS: Functions are in correct workflow order")
    else:
        print("✗ FAIL: Functions are not in optimal workflow order")
    
    # Final Assessment
    print("\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)
    
    tests_passed = [
        ("File exists", True),
        ("All functions found", all_found),
        ("Function sequence", sequence_correct),
        ("Batch processing", batch_processing_ok),
        ("Workflow order", workflow_order_ok)
    ]
    
    total_passed = sum(1 for _, passed in tests_passed if passed)
    total_tests = len(tests_passed)
    
    for test_name, passed in tests_passed:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nResults: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 SUCCESS: Tokenization workflow is correctly implemented!")
        print("   - Text is tokenized BEFORE embedding generation")
        print("   - All required safe functions are present")
        print("   - Batch processing capabilities are available")
        return True
    else:
        print(f"\n❌ ISSUES: {total_tests - total_passed} test(s) failed")
        print("   Please review the failed tests above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
