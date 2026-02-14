// Database types
export interface User {
  id: string;
  email: string;
  password_hash: string;
  name?: string;
  phone?: string;
  credits: number;
  subscription_status: 'none' | 'active' | 'cancelled';
  subscription_expires_at?: string;
  stripe_customer_id?: string;
  created_at: string;
  updated_at: string;
}

export interface Diagnosis {
  id: string;
  user_id: string;
  vehicle_year: number;
  vehicle_make: string;
  vehicle_model: string;
  vin?: string;
  mileage?: number;
  symptom_text: string;
  audio_url?: string;
  photo_urls?: string[];
  audio_features?: AudioFeatures;
  diagnosis_result: DiagnosisResult;
  primary_issue?: string;
  confidence_score?: number;
  severity?: 'low' | 'medium' | 'high' | 'critical';
  safe_to_drive?: boolean;
  retrieved_tsbs?: any[];
  retrieved_procedures?: any[];
  audio_classification?: AudioClassification[];
  repair_steps?: RepairStep[];
  parts_needed?: PartInfo[];
  estimated_cost?: CostEstimate;
  youtube_videos?: YouTubeVideo[];
  safety_warnings?: string[];
  immediate_action_required?: string;
  credits_used: number;
  processing_time_ms?: number;
  created_at: string;
  updated_at: string;
}

export interface Payment {
  id: string;
  user_id: string;
  stripe_payment_intent_id: string;
  amount: number;
  currency: string;
  status: 'pending' | 'succeeded' | 'failed';
  credits_purchased: number;
  payment_method?: string;
  metadata?: any;
  created_at: string;
}

export interface VehicleHistoryLookup {
  id: string;
  user_id: string;
  vin: string;
  nhtsa_data?: any;
  recalls?: any[];
  complaints?: any[];
  vehicle_history?: any;
  accident_history?: any;
  ownership_history?: any;
  title_info?: any;
  estimated_value?: any;
  market_price?: any;
  credits_used: number;
  created_at: string;
}

export interface InsuranceQuote {
  id: string;
  user_id: string;
  vehicle_year: number;
  vehicle_make: string;
  vehicle_model: string;
  vin?: string;
  zip_code?: string;
  age?: number;
  driving_history?: string;
  quotes: InsuranceQuoteResult[];
  affiliate_clicks?: any;
  created_at: string;
}

// AI/RAG types
export interface AudioFeatures {
  spectral_centroid: number;
  spectral_rolloff: number;
  spectral_bandwidth: number;
  spectral_flatness: number;
  mfcc: number[];
  chroma: number[];
  zero_crossing_rate: number;
  rms_energy: number;
  tempo: number;
  harmonic_ratio: number;
  dominant_frequencies: number[];
  peak_frequency: number;
}

export interface AudioClassification {
  class: string;
  confidence: number;
  component: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  likelyCauses?: string[];
  safetyWarning?: string;
}

export interface AudioPattern {
  type: 'constant' | 'rhythmic';
  frequency?: number;
  source?: 'wheel_rotation' | 'engine_rpm' | 'continuous';
}

export interface AudioAnalysis {
  features: AudioFeatures;
  classification: AudioClassification[];
  pattern?: AudioPattern;
  matches?: any[];
  confidence: number;
}

export interface SafetyAlert {
  system: string;
  detected: string;
  action: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  confidence: number;
}

export interface SafetyCheck {
  hasCriticalIssues: boolean;
  alerts: SafetyAlert[];
  safeToDrive: boolean | null;
}

export interface TSB {
  id: string;
  source: string;
  make: string;
  model: string;
  year: number | number[];
  component: string;
  symptom: string;
  diagnosis: string;
  affectedParts?: string[];
  severity: 'low' | 'medium' | 'high' | 'critical';
  driveability: string;
  urgency: string;
  repairProcedure: string;
  partNumbers?: string[];
  laborHours?: number;
  successRate?: number;
  similarity?: number; // From vector search
}

export interface DiagnosisResult {
  primaryDiagnosis: string;
  differential: string[];
  confidence: number;
  reasoning: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  safeToDrive: boolean;
  drivingRestrictions?: string;
  immediateAction?: string;
  repairUrgency: string;
  repairSteps: RepairStep[];
  partsNeeded: PartInfo[];
  estimatedCost: CostEstimate;
  youtubeVideos: YouTubeVideo[];
  safetyWarnings: string[];
  relatedTSBs?: TSB[];
  audioAnalysis?: AudioAnalysis;
  safetyCheck?: SafetyCheck;
}

export interface RepairStep {
  step: number;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'professional';
  estimatedTime: string;
  tools: string[];
  safetyWarnings?: string[];
}

export interface PartInfo {
  name: string;
  oemPartNumber?: string;
  aftermarketOptions?: string[];
  quantity: number;
  estimatedPrice: {
    min: number;
    max: number;
    currency: string;
  };
  required: boolean;
}

export interface CostEstimate {
  parts: {
    min: number;
    max: number;
  };
  labor: {
    min: number;
    max: number;
    hours: number;
  };
  total: {
    min: number;
    max: number;
  };
  currency: string;
  diyPossible: boolean;
  diySavings?: number;
}

export interface YouTubeVideo {
  title: string;
  videoId: string;
  url: string;
  thumbnailUrl: string;
  channelName: string;
  views?: number;
  relevanceScore?: number;
}

// Vehicle API types
export interface VehicleInfo {
  year: number;
  make: string;
  model: string;
  vin?: string;
  mileage?: number;
}

export interface NHTSARecall {
  recallNumber: string;
  component: string;
  summary: string;
  consequence: string;
  remedy: string;
  recallDate: string;
}

export interface VINDecodeResult {
  make: string;
  model: string;
  year: number;
  trim?: string;
  engine?: string;
  transmission?: string;
  drivetrain?: string;
  bodyStyle?: string;
  manufacturerName?: string;
  plantCountry?: string;
}

// Insurance types
export interface InsuranceQuoteResult {
  provider: string;
  monthlyPremium: number;
  coverageLevel: string;
  deductible: number;
  affiliateUrl: string;
  affiliateCode: string;
  features: string[];
  rating: number;
}

export interface InsuranceQuoteRequest {
  vehicle: VehicleInfo;
  zipCode: string;
  age: number;
  drivingHistory: 'clean' | 'minor_violations' | 'major_violations';
  coverageType: 'liability' | 'collision' | 'comprehensive';
}

// API Request/Response types
export interface DiagnosticRequest {
  vehicle: VehicleInfo;
  symptomText: string;
  audioFile?: File | Blob;
  photoFiles?: (File | Blob)[];
}

export interface DiagnosticResponse {
  success: boolean;
  diagnosis?: DiagnosisResult;
  diagnosisId?: string;
  creditsRemaining?: number;
  error?: string;
}

export interface AuthResponse {
  success: boolean;
  token?: string;
  user?: Omit<User, 'password_hash'>;
  error?: string;
}

export interface CreditsResponse {
  success: boolean;
  credits?: number;
  error?: string;
}
