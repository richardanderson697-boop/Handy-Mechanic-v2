import Stripe from 'stripe';

if (!process.env.STRIPE_SECRET_KEY) {
  throw new Error('Missing STRIPE_SECRET_KEY environment variable');
}

export const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
  apiVersion: '2024-12-18.acacia',
  typescript: true,
});

// Credit packages
export const CREDIT_PACKAGES = {
  single: {
    id: 'single',
    name: 'Single Diagnosis',
    credits: 1,
    price: 499, // $4.99 in cents
    description: 'One-time diagnostic analysis',
  },
  pack_3: {
    id: 'pack_3',
    name: '3-Pack',
    credits: 3,
    price: 999, // $9.99 in cents (save $5)
    description: 'Three diagnostic analyses',
    savings: 500, // Save $5
  },
  pack_10: {
    id: 'pack_10',
    name: '10-Pack',
    credits: 10,
    price: 2999, // $29.99 in cents (save $20)
    description: 'Ten diagnostic analyses',
    savings: 2000, // Save $20
  },
};

/**
 * Create a payment intent for purchasing credits
 */
export async function createPaymentIntent(
  userId: string,
  packageId: keyof typeof CREDIT_PACKAGES,
  customerEmail: string
): Promise<Stripe.PaymentIntent> {
  const pkg = CREDIT_PACKAGES[packageId];

  if (!pkg) {
    throw new Error('Invalid package ID');
  }

  const paymentIntent = await stripe.paymentIntents.create({
    amount: pkg.price,
    currency: 'usd',
    metadata: {
      userId,
      packageId,
      credits: pkg.credits.toString(),
    },
    receipt_email: customerEmail,
    description: `Handy Mechanic AI - ${pkg.name}`,
  });

  return paymentIntent;
}

/**
 * Create or retrieve Stripe customer
 */
export async function getOrCreateStripeCustomer(
  userId: string,
  email: string,
  name?: string
): Promise<Stripe.Customer> {
  // Search for existing customer by metadata
  const existingCustomers = await stripe.customers.list({
    email,
    limit: 1,
  });

  if (existingCustomers.data.length > 0) {
    return existingCustomers.data[0];
  }

  // Create new customer
  const customer = await stripe.customers.create({
    email,
    name,
    metadata: {
      userId,
    },
  });

  return customer;
}

/**
 * Create a checkout session for purchasing credits
 */
export async function createCheckoutSession(
  userId: string,
  packageId: keyof typeof CREDIT_PACKAGES,
  customerEmail: string,
  successUrl: string,
  cancelUrl: string
): Promise<Stripe.Checkout.Session> {
  const pkg = CREDIT_PACKAGES[packageId];

  if (!pkg) {
    throw new Error('Invalid package ID');
  }

  // Get or create customer
  const customer = await getOrCreateStripeCustomer(userId, customerEmail);

  const session = await stripe.checkout.sessions.create({
    customer: customer.id,
    payment_method_types: ['card'],
    line_items: [
      {
        price_data: {
          currency: 'usd',
          product_data: {
            name: `Handy Mechanic AI - ${pkg.name}`,
            description: pkg.description,
          },
          unit_amount: pkg.price,
        },
        quantity: 1,
      },
    ],
    mode: 'payment',
    success_url: successUrl,
    cancel_url: cancelUrl,
    metadata: {
      userId,
      packageId,
      credits: pkg.credits.toString(),
    },
  });

  return session;
}

/**
 * Handle successful payment - add credits to user
 */
export async function handleSuccessfulPayment(
  paymentIntentId: string,
  userId: string,
  credits: number
): Promise<void> {
  console.log('[v0] Processing successful payment:', {
    paymentIntentId,
    userId,
    credits,
  });

  // Credits are added by the webhook handler or payment confirmation endpoint
  // This function can be extended for additional business logic
}

/**
 * Verify webhook signature
 */
export function constructWebhookEvent(
  payload: string | Buffer,
  signature: string
): Stripe.Event {
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

  if (!webhookSecret) {
    throw new Error('Missing STRIPE_WEBHOOK_SECRET');
  }

  return stripe.webhooks.constructEvent(payload, signature, webhookSecret);
}
