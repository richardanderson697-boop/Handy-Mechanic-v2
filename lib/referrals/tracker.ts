import { ReferralAction, AffiliatePartner } from '@/types/referrals';

/**
 * Generates an attributed affiliate link for the user.
 * This ID should be stored in your DB to match against partner postbacks.
 */
export const generateAffiliateUrl = (
  partner: AffiliatePartner,
  userId: string,
  diagnosisId: string,
  vin?: string
): string => {
  // Create a unique tracking string (SubID) for the partner
  const trackingId = `ref_${userId}_${Date.now()}`;

  // Build the URL using the partner's template
  // Example: https://endurance.com/quote?aff_id=123&subid=ref_user_123
  let finalUrl = partner.deep_link_template
    .replace('{{trackingId}}', trackingId)
    .replace('{{userId}}', userId);

  if (vin) {
    finalUrl += `&vin=${vin}`;
  }

  return finalUrl;
};

/**
 * Mock function to simulate tracking a click in your DB
 */
export const logReferralClick = async (action: ReferralAction) => {
  console.log(`[LOG]: Referral clicked for ${action.provider_name}. Tracking ID: ${action.id}`);
  // In production, execute: await db.referralAction.create({ data: action });
};
