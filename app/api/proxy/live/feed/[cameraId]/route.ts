import { NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET(
  request: Request,
  { params }: { params: { cameraId: string } }
) {
  try {
    const { cameraId } = params;
    
    const response = await fetch(`${API_BASE_URL}/live/feed/${cameraId}`, {
      headers: {
        ...Object.fromEntries(request.headers.entries()),
        'host': new URL(API_BASE_URL).host,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: errorData.detail || 'Failed to get live feed' },
        { status: response.status }
      );
    }

    // Forward the response as a stream
    const { readable, writable } = new TransformStream();
    response.body?.pipeTo(writable);

    return new NextResponse(readable, {
      headers: {
        'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      },
    });
  } catch (error) {
    console.error('Proxy error in /api/proxy/live/feed/[cameraId]:', error);
    return NextResponse.json(
      { error: 'Failed to connect to the server' },
      { status: 500 }
    );
  }
}
