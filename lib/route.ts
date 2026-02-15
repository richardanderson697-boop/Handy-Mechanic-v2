import { NextRequest, NextResponse } from 'next/server';
import { supabaseAdmin } from '@/lib/database';
import { getUserFromHeaders } from '@/lib/auth';
import {
  decodeVIN,
  getVINRecalls,
  getVehicleComplaints,
  getVehicleHistory,
  estimateVehicleValue,
  isValidVIN,
} from '@/lib/vehicle/vin-decoder';

export async function POST(request: NextRequest) {
  try {
    const tokenPayload = getUserFromHeaders(request.headers);
    if (!tokenPayload) {
      return NextResponse.json(
        { success: false, error: 'Unauthorized' },
        { status: 401 }
      );
    }

    // Get user
    const { data: user } = await supabaseAdmin
      .from('users')
      .select('*')
      .eq('id', tokenPayload.userId)
      .single();

    if (!user) {
      return NextResponse.json(
        { success: false, error: 'User not found' },
        { status: 404 }
      );
    }

    const body = await request.json();
    const { vin, mileage, condition } = body;

    if (!vin) {
      return NextResponse.json(
        { success: false, error: 'VIN is required' },
        { status: 400 }
      );
    }

    if (!isValidVIN(vin)) {
      return NextResponse.json(
        { success: false, error: 'Invalid VIN format' },
        { status: 400 }
      );
    }

    console.log('[v0] Fetching vehicle history for VIN:', vin);

    // Check if we have cached data for this VIN (within last 30 days)
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const { data: cachedHistory } = await supabaseAdmin
      .from('vehicle_history_lookups')
      .select('*')
      .eq('vin', vin)
      .gte('created_at', thirtyDaysAgo.toISOString())
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    if (cachedHistory) {
      console.log('[v0] Returning cached vehicle history');
      return NextResponse.json(
        {
          success: true,
          cached: true,
          history: cachedHistory,
        },
        { status: 200 }
      );
    }

    // Fetch fresh data
    const [vehicleInfo, recalls, vehicleHistory] = await Promise.all([
      decodeVIN(vin),
      getVINRecalls(vin),
      getVehicleHistory(vin),
    ]);

    if (!vehicleInfo) {
      return NextResponse.json(
        { success: false, error: 'Failed to decode VIN' },
        { status: 404 }
      );
    }

    // Get complaints for this vehicle
    const complaints = await getVehicleComplaints(
      vehicleInfo.make,
      vehicleInfo.model,
      vehicleInfo.year
    );

    // Estimate value
    const estimatedValue = await estimateVehicleValue(
      vehicleInfo.make,
      vehicleInfo.model,
      vehicleInfo.year,
      mileage,
      condition
    );

    // Prepare NHTSA data
    const nhtsaData = {
      make: vehicleInfo.make,
      model: vehicleInfo.model,
      year: vehicleInfo.year,
      trim: vehicleInfo.trim,
      engine: vehicleInfo.engine,
      transmission: vehicleInfo.transmission,
      drivetrain: vehicleInfo.drivetrain,
      bodyStyle: vehicleInfo.bodyStyle,
    };

    // Save to database
    const { data: savedHistory, error: saveError } = await supabaseAdmin
      .from('vehicle_history_lookups')
      .insert([
        {
          user_id: user.id,
          vin,
          nhtsa_data: nhtsaData,
          recalls: recalls.length > 0 ? recalls : null,
          complaints: complaints.length > 0 ? complaints : null,
          vehicle_history: vehicleHistory,
          estimated_value: estimatedValue,
          credits_used: 0, // VIN lookups are free
        },
      ])
      .select()
      .single();

    if (saveError) {
      console.error('[v0] Failed to save vehicle history:', saveError);
    }

    console.log('[v0] Vehicle history fetched successfully');

    return NextResponse.json(
      {
        success: true,
        cached: false,
        history: savedHistory || {
          vin,
          nhtsa_data: nhtsaData,
          recalls,
          complaints,
          vehicle_history: vehicleHistory,
          estimated_value: estimatedValue,
        },
      },
      { status: 200 }
    );

  } catch (error) {
    console.error('[v0] Vehicle history error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch vehicle history' },
      { status: 500 }
    );
  }
}

// Get user's vehicle history lookups
export async function GET(request: NextRequest) {
  try {
    const tokenPayload = getUserFromHeaders(request.headers);
    if (!tokenPayload) {
      return NextResponse.json(
        { success: false, error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const { data: histories, error } = await supabaseAdmin
      .from('vehicle_history_lookups')
      .select('*')
      .eq('user_id', tokenPayload.userId)
      .order('created_at', { ascending: false })
      .limit(20);

    if (error) {
      return NextResponse.json(
        { success: false, error: 'Failed to fetch histories' },
        { status: 500 }
      );
    }

    return NextResponse.json(
      { success: true, histories },
      { status: 200 }
    );

  } catch (error) {
    console.error('[v0] Get histories error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch histories' },
      { status: 500 }
    );
  }
}
