// types/affiliate.ts

export interface Referral {
  id: string;
  user_id: string;
  diagnosis_id?: string; // Link to the specific vehicle report that triggered the lead
  type: 'insurance' | 'warranty' | 'financing';
  provider_name: string; // e.g., "Endurance", "LightStream"
  status: 'clicked' | 'applied' | 'converted';
  commission_amount?: number;
  referral_url: string;
  metadata?: {
    loan_amount?: number;
    warranty_plan_type?: string;
    vehicle_vin?: string;
  };
  created_at: string;
  updated_at: string;
}
