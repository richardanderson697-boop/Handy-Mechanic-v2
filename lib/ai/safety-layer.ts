import { DiagnosisResult } from '@/types/diagnosis';

/**
 * Validates AI-generated diagnosis against safety constraints.
 * Ensures high-risk mechanical issues are flagged for professional inspection.
 */
export const validateDiagnosisSafety = (diagnosis: any): DiagnosisResult => {
  const safetyKeywords = ['brake failure', 'steering lock', 'fuel leak', 'fire', 'airbag'];
  const riskKeywords = ['transmission', 'timing belt', 'suspension', 'overheating'];

  let safetyScore = 1.0;
  let recommendation = "Safe to proceed with caution.";

  // 1. Check for Critical Safety Failures
  const hasCriticalIssue = safetyKeywords.some(kw => 
    diagnosis.summary.toLowerCase().includes(kw)
  );

  if (hasCriticalIssue) {
    safetyScore = 0.1;
    recommendation = "CRITICAL SAFETY RISK: Do not drive. Immediate professional inspection required.";
  } 

  // 2. Check for High-Cost/High-Risk Mechanical Issues
  const hasHighRiskIssue = riskKeywords.some(kw => 
    diagnosis.summary.toLowerCase().includes(kw)
  );

  if (hasHighRiskIssue && safetyScore > 0.5) {
    safetyScore = 0.5;
    recommendation = "High-risk mechanical issue detected. Get a professional PPI before purchase.";
  }

  return {
    ...diagnosis,
    safety_score: safetyScore,
    safety_recommendation: recommendation,
    is_safe_to_drive: safetyScore > 0.4
  };
};
