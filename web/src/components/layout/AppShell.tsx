import type { AppPage } from "../../App";

interface Props {
  children: React.ReactNode;
  centerContent?: boolean;
  page: AppPage;
  onNavigate: (page: AppPage) => void;
}

export function AppShell({ children, centerContent = false, page, onNavigate }: Props) {
  return (
    <div className="flex h-screen">
      {/* Main content area — scrollable */}
      <div className="flex flex-1 flex-col overflow-y-auto">
        {/* Top bar */}
        <header className="sticky top-0 z-30 border-b border-elevated bg-canvas/80 backdrop-blur-md" role="banner">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="flex items-center gap-6">
              <button
                onClick={() => onNavigate("home")}
                className="flex items-center gap-2 focus:outline-none"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10" aria-hidden="true">
                  <svg className="h-4 w-4 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
                  </svg>
                </div>
                <h1 className="text-base font-semibold text-text" style={{ fontFamily: "var(--font-heading)" }}>
                  Remedy PDF Desktop
                </h1>
              </button>

              <nav className="flex items-center gap-1" role="navigation" aria-label="Main navigation">
                <NavLink active={page === "home"} onClick={() => onNavigate("home")}>
                  Remediate
                </NavLink>
                <NavLink active={page === "how-it-works"} onClick={() => onNavigate("how-it-works")}>
                  How It Works
                </NavLink>
              </nav>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-text-muted hidden sm:block">Document Accessibility Remediation</span>
              <button
                onClick={() => onNavigate("model-settings")}
                aria-label="Model settings"
                className="rounded-lg p-2 text-text-muted hover:text-text hover:bg-elevated/50 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a6.759 6.759 0 010 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 010-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
            </div>
          </div>
        </header>

        {/* Main */}
        <main
          id="main-content"
          className={`flex flex-1 flex-col px-8 py-8 ${centerContent ? "items-center justify-center" : ""}`}
          role="main"
        >
          {children}
        </main>
      </div>
    </div>
  );
}

function NavLink({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-canvas ${
        active
          ? "bg-primary/10 text-primary"
          : "text-text-muted hover:text-text hover:bg-elevated/50"
      }`}
      aria-current={active ? "page" : undefined}
    >
      {children}
    </button>
  );
}
