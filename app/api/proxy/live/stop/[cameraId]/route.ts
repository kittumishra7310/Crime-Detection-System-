import { NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(
  request: Request,
  { params }: { params: { cameraId: string } }
) {
  try {
    const { cameraId } = params;
    
    const response = await fetch(`${API_BASE_URL}/live/stop/${cameraId}`, {
      method: 'POST',
      headers: {
        ...Object.fromEntries(request.headers.entries()),
        'host': new URL(API_BASE_URL).host,
      },
      body: request.body,
    });

    const data = await response.json().catch(() => ({}));
    
    if (!response.ok) {
      return NextResponse.json(
        { error: data.detail || 'Failed to stop live detection' },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Proxy error in /api/proxy/live/stop/[cameraId]:', error);
    return NextResponse.json(
      { error: 'Failed to connect to the server' },
      { status: 500 }
    );
  }
}
