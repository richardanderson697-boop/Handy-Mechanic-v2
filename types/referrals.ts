// types/referrals.ts

export interface ReferralAction {
  id: string;
  user_id: string;
  diagnosis_id: string; // The report that triggered the need
  category: 'financing' | 'warranty' | 'insurance';
  provider_id: string; // e.g., 'endurance-001'
  status: 'offered' | 'clicked' | 'converted';
  lead_value?: number; // Estimated commission
  external_ref_id?: string; // Partner's tracking ID
  created_at: string;
}

export interface AffiliatePartner {
  id: string;
  name: string;
  category: 'financing' | 'warranty' | 'insurance';
  base_commission: number;
  commission_type: 'CPL' | 'CPA'; // Lead vs. Sale
  deep_link_template: string; // e.g., "https://partner.com?vin={{vin}}&ref={{my_id}}"
}
