import React from 'react';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@/components/ThemeProvider';

jest.mock('next-themes', () => ({
  ThemeProvider: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
    <div data-testid="next-themes-provider" data-props={JSON.stringify(props)}>
      {children}
    </div>
  ),
}));

describe('ThemeProvider', () => {
  it('should render children correctly', () => {
    render(
      <ThemeProvider>
        <div data-testid="child">Content</div>
      </ThemeProvider>
    );

    expect(screen.getByTestId('child')).toBeInTheDocument();
    expect(screen.getByText('Content')).toBeInTheDocument();
  });

  it('should pass all props to NextThemesProvider', () => {
    render(
      <ThemeProvider
        attribute="class"
        defaultTheme="dark"
        enableSystem={false}
        storageKey="my-theme"
      >
        <div>Content</div>
      </ThemeProvider>
    );

    const provider = screen.getByTestId('next-themes-provider');
    const passedProps = JSON.parse(provider.getAttribute('data-props') || '{}');

    expect(passedProps).toEqual({
      attribute: 'class',
      defaultTheme: 'dark',
      enableSystem: false,
      storageKey: 'my-theme',
    });
  });
});