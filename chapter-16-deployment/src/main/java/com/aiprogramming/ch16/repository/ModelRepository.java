package com.aiprogramming.ch16.repository;

import com.aiprogramming.ch16.model.AIModel;
import com.aiprogramming.ch16.model.ModelStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Repository interface for AI model persistence operations
 */
@Repository
public interface ModelRepository extends JpaRepository<AIModel, Long> {

    /**
     * Find model by its unique model ID
     */
    Optional<AIModel> findByModelId(String modelId);

    /**
     * Find models by status
     */
    List<AIModel> findByStatus(ModelStatus status);

    /**
     * Find models by name (partial match)
     */
    List<AIModel> findByNameContainingIgnoreCase(String name);

    /**
     * Find models by version
     */
    List<AIModel> findByVersion(String version);

    /**
     * Find models created by a specific user
     */
    List<AIModel> findByCreatedBy(String createdBy);

    /**
     * Find models created after a specific date
     */
    List<AIModel> findByCreatedAtAfter(LocalDateTime date);

    /**
     * Find models updated after a specific date
     */
    List<AIModel> findByUpdatedAtAfter(LocalDateTime date);

    /**
     * Find active models (deployed or A/B testing)
     */
    @Query("SELECT m FROM AIModel m WHERE m.status IN ('DEPLOYED', 'AB_TESTING')")
    List<AIModel> findActiveModels();

    /**
     * Find models with accuracy above threshold
     */
    @Query("SELECT m FROM AIModel m WHERE m.accuracy >= :minAccuracy")
    List<AIModel> findByAccuracyAbove(@Param("minAccuracy") double minAccuracy);

    /**
     * Find models by type using discriminator
     */
    @Query("SELECT m FROM AIModel m WHERE TYPE(m) = :modelType")
    List<AIModel> findByModelType(@Param("modelType") Class<? extends AIModel> modelType);

    /**
     * Count models by status
     */
    long countByStatus(ModelStatus status);

    /**
     * Check if model exists by model ID
     */
    boolean existsByModelId(String modelId);

    /**
     * Find latest version of a model by name
     */
    @Query("SELECT m FROM AIModel m WHERE m.name = :name ORDER BY m.version DESC")
    List<AIModel> findLatestVersionsByName(@Param("name") String name);

    /**
     * Find models with specific metadata key-value pair
     */
    @Query("SELECT m FROM AIModel m JOIN m.metadata md WHERE KEY(md) = :key AND VALUE(md) = :value")
    List<AIModel> findByMetadataKeyValue(@Param("key") String key, @Param("value") String value);

    /**
     * Find models created in date range
     */
    @Query("SELECT m FROM AIModel m WHERE m.createdAt BETWEEN :startDate AND :endDate")
    List<AIModel> findByCreatedAtBetween(@Param("startDate") LocalDateTime startDate, 
                                        @Param("endDate") LocalDateTime endDate);

    /**
     * Find models with highest accuracy
     */
    @Query("SELECT m FROM AIModel m WHERE m.accuracy IS NOT NULL ORDER BY m.accuracy DESC")
    List<AIModel> findTopModelsByAccuracy();

    /**
     * Find models by name and version
     */
    Optional<AIModel> findByNameAndVersion(String name, String version);
}
