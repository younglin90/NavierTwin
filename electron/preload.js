// contextIsolation 을 유지한 최소 preload. 현재 렌더러(trame 앱)와 별도 IPC 는
// 필요하지 않다. 향후 네이티브 메뉴/파일 다이얼로그 브리지가 필요하면 여기에
// contextBridge.exposeInMainWorld(...) 를 추가한다.
