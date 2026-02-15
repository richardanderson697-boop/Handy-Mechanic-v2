// app/actions/referrals.ts
'use server'

import { db } from '@/lib/db';
import { nanoid } from 'nanoid';

export async function generateReferralLink(type: string, diagnosisId: string) {
  const refId = nanoid(); // Unique tracking ID for the affiliate partner
  
  // Log the intent before redirecting
  await db.referralAction.create({
    data: {
      id: refId,
      diagnosisId,
      type,
      status: 'clicked',
    }
  });

  // Return the partner's deep link with your tracking ID attached
  const partnerLinks = {
    warranty: `https://partner.com/warranty?vin=...&subid=${refId}`,
    finance: `https://lender.com/apply?subid=${refId}`
  };

  return partnerLinks[type];
}
