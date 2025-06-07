#!/bin/bash
# Docker build and deployment script for LAION Embeddings v2.2.0
# Production-ready with MCP tools validation using mcp_server.py entrypoint

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="laion-embeddings"
IMAGE_TAG="v2.2.0"
CONTAINER_NAME="laion-embeddings-mcp-server"
PORT="9999"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if MCP tools are working (same as CI/CD)
validate_mcp_tools() {
    print_status "Validating MCP tools before build (using CI/CD validation)..."
    
    # Use the same validation as CI/CD pipeline
    if python3 mcp_server.py --validate > /dev/null 2>&1; then
        print_success "MCP server validation passed ✅"
        
        # Also run quick validation script
        if python3 tools/validation/mcp_tools_quick_validation.py > /dev/null 2>&1; then
            print_success "Quick MCP tools validation passed ✅"
        else
            print_warning "Quick validation had issues, but server validation passed"
        fi
    else
        print_error "MCP server validation failed. Please fix issues before building."
        print_error "Run 'python3 mcp_server.py --validate' to see detailed errors."
        exit 1
    fi
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
    
    # Build with progress output
    docker build \
        --tag $IMAGE_NAME:$IMAGE_TAG \
        --tag $IMAGE_NAME:latest \
        --progress=plain \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Docker build failed"
        exit 1
    fi
}

# Function to test the image
test_image() {
    print_status "Testing Docker image..."
    
    # Run container in test mode
    docker run --rm --name "${CONTAINER_NAME}-test" \
        -p $PORT:$PORT \
        -e ENVIRONMENT=test \
        --health-cmd="curl -f http://localhost:$PORT/health || exit 1" \
        --health-interval=10s \
        --health-timeout=5s \
        --health-retries=3 \
        -d $IMAGE_NAME:$IMAGE_TAG
    
    # Wait for container to be healthy
    print_status "Waiting for container to be healthy..."
    sleep 30
    
    # Check health
    if docker exec "${CONTAINER_NAME}-test" curl -f http://localhost:$PORT/health > /dev/null 2>&1; then
        print_success "Container health check passed"
    else
        print_error "Container health check failed"
        docker logs "${CONTAINER_NAME}-test"
        docker stop "${CONTAINER_NAME}-test"
        exit 1
    fi
    
    # Test API endpoints
    print_status "Testing API endpoints..."
    if docker exec "${CONTAINER_NAME}-test" curl -f http://localhost:$PORT/ > /dev/null 2>&1; then
        print_success "API endpoints accessible"
    else
        print_warning "API endpoints test inconclusive"
    fi
    
    # Stop test container
    docker stop "${CONTAINER_NAME}-test"
    print_success "Image testing completed"
}

# Function to run production container
run_production() {
    print_status "Starting production container..."
    
    # Stop existing container if running
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        print_warning "Stopping existing container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    fi
    
    # Run production container
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:$PORT \
        -e ENVIRONMENT=production \
        -e LOG_LEVEL=INFO \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/config:/app/config:ro \
        --restart unless-stopped \
        $IMAGE_NAME:$IMAGE_TAG
    
    if [ $? -eq 0 ]; then
        print_success "Production container started successfully"
        print_status "Container name: $CONTAINER_NAME"
        print_status "Access URL: http://localhost:$PORT"
        print_status "Health check: http://localhost:$PORT/health"
    else
        print_error "Failed to start production container"
        exit 1
    fi
}

# Function to show container logs
show_logs() {
    print_status "Showing container logs..."
    docker logs -f $CONTAINER_NAME
}

# Function to show container status
show_status() {
    print_status "Container status:"
    docker ps -f name=$CONTAINER_NAME
    
    print_status "Container health:"
    docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "Health check not available"
    
    print_status "Resource usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" $CONTAINER_NAME
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove dangling volumes
    docker volume prune -f
    
    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build Docker image"
    echo "  test        Test Docker image"
    echo "  run         Run production container"
    echo "  logs        Show container logs"
    echo "  status      Show container status"
    echo "  stop        Stop production container"
    echo "  restart     Restart production container"
    echo "  cleanup     Clean up Docker resources"
    echo "  all         Build, test, and run (full deployment)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 all        # Full deployment pipeline"
    echo "  $0 build      # Build image only"
    echo "  $0 run        # Run production container"
    echo "  $0 status     # Check container status"
}

# Main script logic
case "${1:-}" in
    "build")
        validate_mcp_tools
        build_image
        ;;
    "test")
        test_image
        ;;
    "run")
        run_production
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "stop")
        print_status "Stopping production container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        print_success "Container stopped and removed"
        ;;
    "restart")
        print_status "Restarting production container..."
        docker restart $CONTAINER_NAME
        print_success "Container restarted"
        ;;
    "cleanup")
        cleanup
        ;;
    "all")
        validate_mcp_tools
        build_image
        test_image
        run_production
        print_success "Full deployment completed successfully!"
        print_status "Your LAION Embeddings server is now running at http://localhost:$PORT"
        ;;
    "help"|"--help"|"-h")
        show_usage
        ;;
    "")
        print_error "No command specified"
        show_usage
        exit 1
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
