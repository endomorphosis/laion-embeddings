# Pull Request Template

## 📋 Description
<!-- Provide a brief description of the changes in this PR -->

## 🔧 Type of Change
<!-- Mark the relevant option with an "x" -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test coverage improvement

## ✅ Testing
<!-- Describe the tests that you ran to verify your changes -->

### Test Results
- [ ] All existing tests pass (`python run_comprehensive_tests.py`)
- [ ] New tests added for new functionality
- [ ] Test coverage maintained or improved

### Test Summary
- Vector Service: ___/23 tests passing
- IPFS Vector Service: ___/15 tests passing  
- Clustering Service: ___/19 tests passing
- Integration Tests: ___/2 tests passing
- Overall: ___/64 tests passing

## 📋 Checklist
<!-- Mark completed items with an "x" -->

### Code Quality
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] I have run linting tools (black, isort, flake8)

### Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have run the comprehensive test suite with 100% success

### Documentation
- [ ] I have made corresponding changes to the documentation
- [ ] I have updated the README if necessary
- [ ] I have updated the API documentation if applicable

## 🔍 Related Issues
<!-- Link any related issues -->
Fixes #(issue number)

## 📸 Screenshots
<!-- If applicable, add screenshots to help explain your changes -->

## 🎯 Production Readiness
<!-- Confirm production readiness -->
- [ ] This change maintains the 100% test success rate
- [ ] This change is compatible with production deployment
- [ ] This change has been tested with the current validated infrastructure
- [ ] No breaking changes that would affect existing deployments

## ⚡ Performance Impact
<!-- Describe any performance implications -->
- [ ] No performance impact
- [ ] Performance improvement
- [ ] Performance regression (explain why acceptable)

## 🔒 Security Considerations
<!-- Describe any security implications -->
- [ ] No security impact
- [ ] Security improvement
- [ ] Potential security concerns (addressed how?)

## 📝 Additional Notes
<!-- Add any additional notes about the PR -->
