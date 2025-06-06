# Updated _register_tools method for src/mcp_server/main.py

async def _register_tools(self):
    """Register all available tools with actual service instances."""
    if not self.tool_registry:
        raise InternalError("Tool registry not initialized")
    
    try:
        # Import service factory
        from .service_factory import ServiceFactory, create_default_service_configs
        
        # Initialize services first
        logger.info("Initializing services for MCP tools...")
        service_configs = create_default_service_configs()
        
        self.service_factory = ServiceFactory(service_configs)
        services = await self.service_factory.initialize_services()
        
        # Embedding tools with actual embedding service
        logger.info("Registering embedding tools...")
        embedding_service = services.get('embedding')
        
        embedding_gen = EmbeddingGenerationTool(embedding_service)
        batch_embedding = BatchEmbeddingTool(embedding_service)
        multimodal_embedding = MultimodalEmbeddingTool(embedding_service)
        
        await self.tool_registry.register_tool(embedding_gen)
        await self.tool_registry.register_tool(batch_embedding)
        await self.tool_registry.register_tool(multimodal_embedding)
        
        # Search tools with vector service
        logger.info("Registering search tools...")
        vector_service = services.get('vector')
        
        semantic_search = SemanticSearchTool(vector_service)
        similarity_search = SimilaritySearchTool(vector_service)
        faceted_search = FacetedSearchTool(vector_service)
        
        await self.tool_registry.register_tool(semantic_search)
        await self.tool_registry.register_tool(similarity_search)
        await self.tool_registry.register_tool(faceted_search)
        
        # Storage tools with vector and IPFS services
        logger.info("Registering storage tools...")
        ipfs_service = services.get('ipfs')
        
        storage_mgmt = StorageManagementTool(vector_service)
        collection_mgmt = CollectionManagementTool(vector_service)
        retrieval = RetrievalTool(vector_service)
        
        await self.tool_registry.register_tool(storage_mgmt)
        await self.tool_registry.register_tool(collection_mgmt)
        await self.tool_registry.register_tool(retrieval)
        
        # Analysis tools with clustering service
        logger.info("Registering analysis tools...")
        clustering_service = services.get('clustering')
        
        cluster_analysis = ClusterAnalysisTool(clustering_service)
        quality_assessment = QualityAssessmentTool(clustering_service)
        dimensionality_reduction = DimensionalityReductionTool(clustering_service)
        
        await self.tool_registry.register_tool(cluster_analysis)
        await self.tool_registry.register_tool(quality_assessment)
        await self.tool_registry.register_tool(dimensionality_reduction)
        
        logger.info(f"Registered {len(self.tool_registry.tools)} tools with actual services")
        
    except Exception as e:
        logger.error(f"Failed to register tools: {e}")
        raise InternalError(f"Tool registration failed: {e}")

# Updated _shutdown_components method for src/mcp_server/main.py

async def _shutdown_components(self):
    """Shutdown all components gracefully."""
    logger.info("Shutting down components...")
    
    try:
        # Shutdown services first
        if hasattr(self, 'service_factory'):
            await self.service_factory.shutdown_services()
            logger.info("Services shutdown complete")
        
        # Shutdown MCP server
        if self.mcp_server:
            await self.mcp_server.shutdown()
            logger.info("MCP server shutdown complete")
        
        # Shutdown session manager
        if self.session_manager:
            await self.session_manager.shutdown()
            logger.info("Session manager shutdown complete")
        
        # Shutdown metrics collector
        if self.metrics_collector:
            self.metrics_collector.shutdown()
            logger.info("Metrics collector shutdown complete")
        
        logger.info("All components shutdown successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        log_error(InternalError(f"Shutdown error: {e}"))
