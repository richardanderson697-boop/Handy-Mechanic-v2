import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import type {
  VehicleInfo,
  DiagnosisResult,
  AudioAnalysis,
  AudioFeatures,
  SafetyCheck,
  TSB,
} from '../types';

export class EnhancedDiagnosticRAG {
  private pinecone: Pinecone;
  private openai: OpenAI;
  private anthropic: Anthropic;
  private index: any;

  constructor() {
    // Initialize Pinecone
    this.pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    // Initialize OpenAI for embeddings
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
    });

    // Initialize Anthropic for diagnosis generation
    this.anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY!,
    });

    // Get index
    const indexName = process.env.PINECONE_INDEX_NAME || 'automotive-diagnostics';
    this.index = this.pinecone.index(indexName);
  }

  /**
   * Main diagnosis function with enterprise-grade RAG
   */
  async diagnose(
    vehicleData: VehicleInfo,
    symptomText: string,
    audioAnalysis?: AudioAnalysis | null,
    photoUrls?: string[]
  ): Promise<DiagnosisResult> {
    try {
      console.log('[v0] Starting RAG diagnosis for:', vehicleData);

      // 1. Build comprehensive query
      const queryText = this.buildQueryText(vehicleData, symptomText, audioAnalysis);
      console.log('[v0] Built query text, length:', queryText.length);

      // 2. Generate embedding for semantic search
      const queryEmbedding = await this.createEmbedding(queryText);
      console.log('[v0] Generated query embedding');

      // 3. Retrieve relevant knowledge from vector database
      const relevantKnowledge = await this.retrieveKnowledge(
        queryEmbedding,
        vehicleData,
        audioAnalysis
      );
      console.log('[v0] Retrieved knowledge:', relevantKnowledge.length, 'results');

      // 4. Check for safety-critical issues FIRST
      const safetyCritical = this.checkSafetyCritical(symptomText, audioAnalysis);
      console.log('[v0] Safety check complete:', safetyCritical.hasCriticalIssues);

      // 5. Generate diagnosis with Claude using retrieved context
      const diagnosis = await this.generateDiagnosis(
        vehicleData,
        symptomText,
        audioAnalysis,
        relevantKnowledge,
        safetyCritical,
        photoUrls
      );

      // 6. Validate and enhance diagnosis
      const validated = this.validateDiagnosis(diagnosis, relevantKnowledge);

      // 7. Add final safety assessment
      const final = this.addSafetyAssessment(validated, safetyCritical);

      console.log('[v0] Diagnosis complete with confidence:', final.confidence);
      return final;

    } catch (error) {
      console.error('[v0] Diagnosis error:', error);
      throw new Error('Diagnosis failed - please try again or consult a mechanic');
    }
  }

  /**
   * Build comprehensive query text for embedding
   */
  private buildQueryText(
    vehicleData: VehicleInfo,
    symptomText: string,
    audioAnalysis?: AudioAnalysis | null
  ): string {
    let query = `
      Vehicle: ${vehicleData.year} ${vehicleData.make} ${vehicleData.model}
      Symptom: ${symptomText}
    `;

    if (vehicleData.mileage) {
      query += `\nMileage: ${vehicleData.mileage} miles`;
    }

    if (audioAnalysis && audioAnalysis.classification.length > 0) {
      const topClass = audioAnalysis.classification[0];
      query += `
      Audio characteristics:
      - Sound type: ${topClass.class}
      - Component: ${topClass.component}
      - Confidence: ${topClass.confidence}
      - Peak frequency: ${audioAnalysis.features.peak_frequency} Hz
      `;
    }

    return query.trim();
  }

  /**
   * Create embedding for semantic search using OpenAI
   */
  private async createEmbedding(text: string): Promise<number[]> {
    try {
      const response = await this.openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
      });

      return response.data[0].embedding;
    } catch (error) {
      console.error('[v0] Embedding creation error:', error);
      throw new Error('Failed to create embedding');
    }
  }

  /**
   * Retrieve relevant knowledge from Pinecone vector database
   */
  private async retrieveKnowledge(
    queryEmbedding: number[],
    vehicleData: VehicleInfo,
    audioAnalysis?: AudioAnalysis | null
  ): Promise<TSB[]> {
    try {
      // Query vector database with vehicle filters
      const yearRange = 2; // +/- 2 years
      
      const queryResponse = await this.index.query({
        vector: queryEmbedding,
        topK: 10,
        includeMetadata: true,
        filter: {
          make: { $eq: vehicleData.make.toLowerCase() },
          model: { $eq: vehicleData.model.toLowerCase() },
          year: {
            $gte: vehicleData.year - yearRange,
            $lte: vehicleData.year + yearRange,
          },
        },
      });

      // Also query without year filter for broader results
      const broadQueryResponse = await this.index.query({
        vector: queryEmbedding,
        topK: 5,
        includeMetadata: true,
        filter: {
          make: { $eq: vehicleData.make.toLowerCase() },
          model: { $eq: vehicleData.model.toLowerCase() },
        },
      });

      // Combine and deduplicate results
      const allMatches = [
        ...queryResponse.matches,
        ...broadQueryResponse.matches,
      ];

      const uniqueMatches = Array.from(
        new Map(allMatches.map(match => [match.id, match])).values()
      );

      // Convert to TSB format with similarity scores
      const tsbs: TSB[] = uniqueMatches
        .filter(match => match.score && match.score > 0.7) // Only high-confidence matches
        .map(match => ({
          id: match.id as string,
          source: match.metadata?.source as string || 'Unknown',
          make: match.metadata?.make as string,
          model: match.metadata?.model as string,
          year: match.metadata?.year as number,
          component: match.metadata?.component as string,
          symptom: match.metadata?.symptom as string,
          diagnosis: match.metadata?.diagnosis as string,
          severity: match.metadata?.severity as any,
          driveability: match.metadata?.driveability as string,
          urgency: match.metadata?.urgency as string,
          repairProcedure: match.metadata?.repairProcedure as string,
          partNumbers: match.metadata?.partNumbers as string[],
          laborHours: match.metadata?.laborHours as number,
          successRate: match.metadata?.successRate as number,
          similarity: match.score,
        }))
        .sort((a, b) => (b.similarity || 0) - (a.similarity || 0));

      return tsbs.slice(0, 10); // Top 10 results

    } catch (error) {
      console.error('[v0] Knowledge retrieval error:', error);
      // Return empty array if Pinecone is not set up yet
      return [];
    }
  }

  /**
   * Check for safety-critical issues
   */
  private checkSafetyCritical(
    symptomText: string,
    audioAnalysis?: AudioAnalysis | null
  ): SafetyCheck {
    const alerts = [];
    const lowerSymptom = symptomText.toLowerCase();

    // Critical brake issues
    if (
      lowerSymptom.includes('no brakes') ||
      lowerSymptom.includes('brake pedal to floor') ||
      lowerSymptom.includes('brakes failed')
    ) {
      alerts.push({
        system: 'brakes',
        detected: 'Brake failure indicators',
        action: 'STOP DRIVING IMMEDIATELY - Complete brake failure risk',
        severity: 'CRITICAL' as const,
        confidence: 0.95,
      });
    }

    // Grinding brakes (from audio or text)
    if (
      lowerSymptom.includes('grinding') &&
      (lowerSymptom.includes('brake') || lowerSymptom.includes('stop'))
    ) {
      alerts.push({
        system: 'brakes',
        detected: 'Grinding brake noise',
        action: 'STOP DRIVING - Brake pad worn to metal, rotor damage likely',
        severity: 'CRITICAL' as const,
        confidence: 0.90,
      });
    }

    // Audio-detected critical issues
    if (audioAnalysis && audioAnalysis.classification.length > 0) {
      for (const classification of audioAnalysis.classification) {
        if (
          classification.class === 'grinding_brakes' &&
          classification.confidence > 0.8
        ) {
          alerts.push({
            system: 'brakes',
            detected: 'Grinding brake noise (audio confirmed)',
            action: 'STOP DRIVING - Brake system failure risk',
            severity: 'CRITICAL' as const,
            confidence: classification.confidence,
          });
        }

        if (
          classification.class === 'engine_knock' &&
          classification.confidence > 0.75
        ) {
          alerts.push({
            system: 'engine',
            detected: 'Engine knocking (audio confirmed)',
            action: 'STOP DRIVING - Engine bearing failure or severe damage risk',
            severity: 'CRITICAL' as const,
            confidence: classification.confidence,
          });
        }
      }
    }

    // Steering issues
    if (
      lowerSymptom.includes('steering locked') ||
      lowerSymptom.includes("can't steer") ||
      lowerSymptom.includes('steering wheel won\'t turn')
    ) {
      alerts.push({
        system: 'steering',
        detected: 'Steering system failure',
        action: 'DO NOT DRIVE - Steering system critical failure',
        severity: 'CRITICAL' as const,
        confidence: 0.95,
      });
    }

    // Fire/smoke
    if (
      lowerSymptom.includes('smoke from engine') ||
      lowerSymptom.includes('burning smell') ||
      lowerSymptom.includes('fire') ||
      lowerSymptom.includes('flames')
    ) {
      alerts.push({
        system: 'fire',
        detected: 'Fire or overheat indicators',
        action: 'PULL OVER IMMEDIATELY - Fire risk',
        severity: 'CRITICAL' as const,
        confidence: 0.95,
      });
    }

    const hasCriticalIssues = alerts.some(a => a.severity === 'CRITICAL');

    return {
      hasCriticalIssues,
      alerts,
      safeToDrive: hasCriticalIssues ? false : null,
    };
  }

  /**
   * Generate diagnosis using Claude with retrieved context
   */
  private async generateDiagnosis(
    vehicleData: VehicleInfo,
    symptomText: string,
    audioAnalysis: AudioAnalysis | null | undefined,
    relevantKnowledge: TSB[],
    safetyCritical: SafetyCheck,
    photoUrls?: string[]
  ): Promise<DiagnosisResult> {
    // Build context from retrieved knowledge
    const knowledgeContext = relevantKnowledge.length > 0
      ? relevantKnowledge
          .map(
            (tsb, idx) => `
[TSB ${idx + 1}] ${tsb.make} ${tsb.model} ${tsb.year}
Component: ${tsb.component}
Symptom: ${tsb.symptom}
Diagnosis: ${tsb.diagnosis}
Severity: ${tsb.severity}
Repair: ${tsb.repairProcedure}
Driveability: ${tsb.driveability}
Similarity: ${((tsb.similarity || 0) * 100).toFixed(1)}%
`
          )
          .join('\n---\n')
      : 'No exact matches found in technical service bulletin database.';

    // Build audio context
    const audioContext = audioAnalysis
      ? `
Audio Analysis Results:
- Classification: ${audioAnalysis.classification.map(c => `${c.class} (${(c.confidence * 100).toFixed(1)}% confidence, ${c.severity} severity)`).join(', ')}
- Peak Frequency: ${audioAnalysis.features.peak_frequency} Hz
- Pattern: ${audioAnalysis.pattern?.type || 'unknown'}
- Overall Confidence: ${(audioAnalysis.confidence * 100).toFixed(1)}%
`
      : 'No audio recording provided.';

    // Build safety context
    const safetyContext = safetyCritical.hasCriticalIssues
      ? `
CRITICAL SAFETY ALERTS:
${safetyCritical.alerts.map(alert => `- ${alert.detected}: ${alert.action}`).join('\n')}
`
      : '';

    const prompt = `You are an expert automotive diagnostic AI assistant with access to a comprehensive knowledge base of Technical Service Bulletins (TSBs), repair manuals, and diagnostic procedures.

VEHICLE INFORMATION:
- Year: ${vehicleData.year}
- Make: ${vehicleData.make}
- Model: ${vehicleData.model}
${vehicleData.mileage ? `- Mileage: ${vehicleData.mileage} miles` : ''}
${vehicleData.vin ? `- VIN: ${vehicleData.vin}` : ''}

REPORTED SYMPTOMS:
${symptomText}

${safetyContext}

${audioContext}

RELEVANT TECHNICAL SERVICE BULLETINS AND KNOWLEDGE BASE:
${knowledgeContext}

Based on the above information, provide a comprehensive diagnostic analysis in the following JSON format:

{
  "primaryDiagnosis": "Clear, specific diagnosis of the most likely issue",
  "differential": ["Alternative diagnosis 1", "Alternative diagnosis 2", "Alternative diagnosis 3"],
  "confidence": 0.85,
  "reasoning": "Detailed explanation of why this is the likely diagnosis, referencing TSBs and audio analysis",
  "severity": "low|medium|high|critical",
  "safeToDrive": true|false,
  "drivingRestrictions": "Specific restrictions if vehicle is driveable (or omit if not driveable)",
  "immediateAction": "Required immediate action (or omit if not critical)",
  "repairUrgency": "address_immediately|within_24_hours|within_week|schedule_soon|monitor",
  "repairSteps": [
    {
      "step": 1,
      "title": "Step title",
      "description": "Detailed description",
      "difficulty": "beginner|intermediate|advanced|professional",
      "estimatedTime": "30 minutes",
      "tools": ["Tool 1", "Tool 2"],
      "safetyWarnings": ["Warning 1"]
    }
  ],
  "partsNeeded": [
    {
      "name": "Part name",
      "oemPartNumber": "12345",
      "aftermarketOptions": ["Brand 1", "Brand 2"],
      "quantity": 1,
      "estimatedPrice": {"min": 50, "max": 100, "currency": "usd"},
      "required": true
    }
  ],
  "estimatedCost": {
    "parts": {"min": 100, "max": 200},
    "labor": {"min": 150, "max": 300, "hours": 2},
    "total": {"min": 250, "max": 500},
    "currency": "usd",
    "diyPossible": true,
    "diySavings": 200
  },
  "safetyWarnings": ["Warning 1", "Warning 2"]
}

IMPORTANT GUIDELINES:
1. If safety-critical issues detected, set safeToDrive to false and provide immediate action
2. Be specific and actionable in repair steps
3. Include realistic cost estimates
4. Reference TSB information when available
5. Consider audio analysis results for confidence scoring
6. Always err on the side of caution for safety
7. If confidence is low (<0.65), recommend professional inspection`;

    try {
      const response = await this.anthropic.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 4000,
        messages: [
          {
            role: 'user',
            content: prompt,
          },
        ],
      });

      // Extract JSON from response
      const content = response.content[0];
      const responseText = content.type === 'text' ? content.text : '';
      
      // Try to parse JSON from response
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('Failed to extract JSON from Claude response');
      }

      const diagnosisData = JSON.parse(jsonMatch[0]);

      // Add YouTube videos based on diagnosis
      const youtubeVideos = await this.searchYouTubeVideos(
        vehicleData,
        diagnosisData.primaryDiagnosis
      );

      return {
        ...diagnosisData,
        youtubeVideos,
        relatedTSBs: relevantKnowledge.slice(0, 5),
        audioAnalysis,
        safetyCheck: safetyCritical,
      };

    } catch (error) {
      console.error('[v0] Claude diagnosis generation error:', error);
      throw new Error('Failed to generate diagnosis');
    }
  }

  /**
   * Search YouTube for relevant repair videos
   */
  private async searchYouTubeVideos(
    vehicleData: VehicleInfo,
    diagnosis: string
  ): Promise<any[]> {
    if (!process.env.YOUTUBE_API_KEY) {
      return [];
    }

    try {
      const query = `${vehicleData.year} ${vehicleData.make} ${vehicleData.model} ${diagnosis} repair how to`;
      
      const response = await fetch(
        `https://www.googleapis.com/youtube/v3/search?part=snippet&q=${encodeURIComponent(query)}&type=video&maxResults=5&key=${process.env.YOUTUBE_API_KEY}`
      );

      if (!response.ok) {
        return [];
      }

      const data = await response.json();

      return data.items?.map((item: any) => ({
        title: item.snippet.title,
        videoId: item.id.videoId,
        url: `https://www.youtube.com/watch?v=${item.id.videoId}`,
        thumbnailUrl: item.snippet.thumbnails.medium.url,
        channelName: item.snippet.channelTitle,
      })) || [];

    } catch (error) {
      console.error('[v0] YouTube search error:', error);
      return [];
    }
  }

  /**
   * Validate diagnosis against knowledge base
   */
  private validateDiagnosis(
    diagnosis: DiagnosisResult,
    knowledge: TSB[]
  ): DiagnosisResult {
    // If we have matching TSBs with high similarity, boost confidence
    if (knowledge.length > 0 && knowledge[0].similarity && knowledge[0].similarity > 0.85) {
      diagnosis.confidence = Math.min(diagnosis.confidence + 0.1, 1.0);
    }

    // If no TSBs found and confidence is already low, reduce further
    if (knowledge.length === 0 && diagnosis.confidence < 0.7) {
      diagnosis.confidence = Math.max(diagnosis.confidence - 0.1, 0.3);
      diagnosis.safetyWarnings.push(
        'Limited technical data available - recommend professional inspection'
      );
    }

    return diagnosis;
  }

  /**
   * Add safety assessment to final diagnosis
   */
  private addSafetyAssessment(
    diagnosis: DiagnosisResult,
    safetyCheck: SafetyCheck
  ): DiagnosisResult {
    // Override diagnosis if critical safety issues detected
    if (safetyCheck.hasCriticalIssues) {
      diagnosis.safeToDrive = false;
      diagnosis.severity = 'critical';
      diagnosis.immediateAction = safetyCheck.alerts[0].action;
      
      // Add all safety alerts to warnings
      safetyCheck.alerts.forEach(alert => {
        if (!diagnosis.safetyWarnings.includes(alert.action)) {
          diagnosis.safetyWarnings.unshift(alert.action);
        }
      });
    }

    return diagnosis;
  }
}
