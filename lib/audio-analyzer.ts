import type { AudioFeatures, AudioClassification, AudioPattern, AudioAnalysis } from '../types';

/**
 * Audio Analysis Engine for Automotive Sounds
 * Processes audio recordings to identify mechanical issues
 */
export class AudioAnalyzer {
  /**
   * Analyze audio file and extract features + classification
   */
  async analyzeAudio(audioBuffer: Buffer): Promise<AudioAnalysis> {
    try {
      console.log('[v0] Starting audio analysis, buffer size:', audioBuffer.length);

      // Extract features from audio
      const features = await this.extractFeatures(audioBuffer);
      console.log('[v0] Extracted audio features');

      // Classify sound based on features
      const classification = this.classifySound(features);
      console.log('[v0] Classified sound:', classification[0]?.class);

      // Analyze temporal pattern
      const pattern = this.analyzePattern(features);

      // Calculate overall confidence
      const confidence = this.calculateConfidence(classification, features);

      return {
        features,
        classification,
        pattern,
        confidence,
      };
    } catch (error) {
      console.error('[v0] Audio analysis error:', error);
      throw new Error('Audio analysis failed');
    }
  }

  /**
   * Extract acoustic features from audio buffer
   * In production, this would use librosa via Python subprocess or Web Audio API
   * For now, we'll create a simplified version that analyzes basic characteristics
   */
  private async extractFeatures(audioBuffer: Buffer): Promise<AudioFeatures> {
    // In a production environment, you would:
    // 1. Convert buffer to audio samples
    // 2. Apply FFT (Fast Fourier Transform)
    // 3. Extract spectral features, MFCCs, etc.
    // 4. Use librosa Python library or Web Audio API

    // For demonstration, we'll return reasonable default features
    // In production, replace this with actual audio processing
    
    return {
      spectral_centroid: 1500, // Hz - brightness of sound
      spectral_rolloff: 3000, // Hz - frequency below which 85% of energy is contained
      spectral_bandwidth: 2000, // Hz - width of frequency range
      spectral_flatness: 0.3, // 0-1, how noise-like vs tone-like
      mfcc: Array(20).fill(0).map(() => Math.random() * 10), // Mel-frequency cepstral coefficients
      chroma: Array(12).fill(0).map(() => Math.random()), // Pitch class profile
      zero_crossing_rate: 0.1, // Rate of sign changes
      rms_energy: 0.4, // Root mean square energy
      tempo: 60, // Beats per minute (for rhythmic sounds)
      harmonic_ratio: 0.6, // Ratio of harmonic to percussive content
      dominant_frequencies: [800, 1600, 2400], // Hz
      peak_frequency: 1200, // Hz
    };
  }

  /**
   * Classify automotive sound based on features
   */
  private classifySound(features: AudioFeatures): AudioClassification[] {
    const classifications: AudioClassification[] = [];

    // Rule-based classification system
    // In production, this would be an ML model (TensorFlow/PyTorch)

    // High-frequency whine (alternator, power steering)
    if (features.spectral_centroid > 2000 && features.peak_frequency > 1500) {
      classifications.push({
        class: 'alternator_whine',
        confidence: 0.80,
        component: 'Alternator bearing',
        severity: 'medium',
        likelyCauses: ['Worn alternator bearing', 'Belt tension issue'],
      });
    }

    // Grinding noise (brakes, bearings)
    if (features.spectral_rolloff > 3000 && features.rms_energy > 0.5) {
      classifications.push({
        class: 'grinding_brakes',
        confidence: 0.85,
        component: 'Brake pads/rotors',
        severity: 'critical',
        likelyCauses: ['Brake pads worn to metal', 'Damaged rotor'],
        safetyWarning: 'CRITICAL: Immediate inspection required - brake failure risk',
      });
    }

    // Belt squeal
    if (
      features.spectral_centroid > 1500 &&
      features.spectral_centroid < 3000 &&
      features.harmonic_ratio > 0.7
    ) {
      classifications.push({
        class: 'belt_squeal',
        confidence: 0.75,
        component: 'Serpentine belt',
        severity: 'low',
        likelyCauses: ['Worn belt', 'Improper tension', 'Belt glazing'],
      });
    }

    // Engine knock/ping
    if (features.zero_crossing_rate > 0.15 && features.tempo > 100) {
      classifications.push({
        class: 'engine_knock',
        confidence: 0.70,
        component: 'Engine internals',
        severity: 'high',
        likelyCauses: ['Pre-ignition', 'Rod bearing failure', 'Low octane fuel'],
        safetyWarning: 'Engine damage risk - reduce speed and have inspected',
      });
    }

    // Wheel bearing rumble
    if (
      features.spectral_centroid < 800 &&
      features.rms_energy > 0.3 &&
      features.spectral_flatness < 0.4
    ) {
      classifications.push({
        class: 'wheel_bearing',
        confidence: 0.78,
        component: 'Wheel bearing',
        severity: 'high',
        likelyCauses: ['Worn wheel bearing'],
        safetyWarning: 'Wheel bearing failure can cause wheel separation',
      });
    }

    // CV joint click
    if (features.tempo > 40 && features.tempo < 80 && features.zero_crossing_rate > 0.1) {
      classifications.push({
        class: 'cv_joint_click',
        confidence: 0.72,
        component: 'CV joint',
        severity: 'medium',
        likelyCauses: ['Worn CV joint', 'Torn CV boot'],
      });
    }

    // Exhaust leak
    if (
      features.spectral_centroid < 600 &&
      features.rms_energy > 0.4 &&
      features.harmonic_ratio < 0.5
    ) {
      classifications.push({
        class: 'exhaust_leak',
        confidence: 0.68,
        component: 'Exhaust system',
        severity: 'low',
        likelyCauses: ['Exhaust manifold leak', 'Muffler damage', 'Pipe corrosion'],
      });
    }

    // If no strong classifications, add "normal operation"
    if (classifications.length === 0) {
      classifications.push({
        class: 'normal_operation',
        confidence: 0.60,
        component: 'General',
        severity: 'low',
        likelyCauses: ['Normal operation noise'],
      });
    }

    // Sort by confidence
    return classifications.sort((a, b) => b.confidence - a.confidence);
  }

  /**
   * Analyze temporal patterns in the sound
   */
  private analyzePattern(features: AudioFeatures): AudioPattern {
    // Determine if sound is rhythmic (related to rotation) or constant

    if (features.tempo > 30 && features.tempo < 120) {
      // Rhythmic pattern detected
      const frequency = features.tempo / 60; // Convert BPM to Hz

      if (frequency > 0.5 && frequency < 2) {
        // Wheel rotation speed (0.5-2 Hz for typical driving)
        return {
          type: 'rhythmic',
          frequency,
          source: 'wheel_rotation',
        };
      } else if (frequency > 15 && frequency < 100) {
        // Engine RPM (900-6000 RPM = 15-100 Hz)
        return {
          type: 'rhythmic',
          frequency,
          source: 'engine_rpm',
        };
      }
    }

    // Constant sound
    return {
      type: 'constant',
      source: 'continuous',
    };
  }

  /**
   * Calculate overall confidence in audio analysis
   */
  private calculateConfidence(
    classifications: AudioClassification[],
    features: AudioFeatures
  ): number {
    if (classifications.length === 0) {
      return 0.3;
    }

    let confidence = classifications[0].confidence;

    // If top 2 classifications are very close, reduce confidence
    if (classifications.length > 1) {
      const diff = classifications[0].confidence - classifications[1].confidence;
      if (diff < 0.15) {
        confidence *= 0.85;
      }
    }

    // Boost confidence if audio quality is good (high RMS energy)
    if (features.rms_energy > 0.6) {
      confidence *= 1.05;
    }

    // Reduce confidence if audio is very noisy (high spectral flatness)
    if (features.spectral_flatness > 0.7) {
      confidence *= 0.9;
    }

    return Math.min(Math.max(confidence, 0.3), 0.95);
  }

  /**
   * Create text description of audio for embedding
   */
  createAudioDescription(analysis: AudioAnalysis): string {
    const topClass = analysis.classification[0];
    if (!topClass) {
      return 'No clear audio signature detected';
    }

    return `
      Sound type: ${topClass.class}
      Component: ${topClass.component}
      Severity: ${topClass.severity}
      Confidence: ${(topClass.confidence * 100).toFixed(1)}%
      Peak frequency: ${analysis.features.peak_frequency} Hz
      Pattern: ${analysis.pattern?.type || 'unknown'}
    `.trim();
  }
}

/**
 * Helper function to convert audio file to buffer
 */
export async function audioFileToBuffer(file: File | Blob): Promise<Buffer> {
  const arrayBuffer = await file.arrayBuffer();
  return Buffer.from(arrayBuffer);
}
