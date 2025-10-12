import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = 'http://localhost:8000';

export async function GET(req: NextRequest) {
  const path = req.nextUrl.pathname.replace('/api', ''); // aget the full path after /api
  const backendUrl = `${API_BASE_URL}${path}`;

  try {
    const response = await fetch(backendUrl, {
      method: 'GET',
      headers: {
        'Accept': 'multipart/x-mixed-replace',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      const errorText = await response.text();
      return new NextResponse(`Backend error: ${errorText}`, { status: response.status });
    }

    const stream = response.body;
    if (!stream) {
      return new NextResponse('No stream from backend', { status: 500 });
    }

    return new NextResponse(stream, {
      headers: {
        'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      },
    });

  } catch (error: any) {
    console.error('Live feed proxy error:', error);
    return new NextResponse(`Proxy error: ${error.message}`, { status: 500 });
  }
}
