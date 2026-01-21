import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Home from '../app/page';

jest.mock('../components/AuthForm', () => {
  return function MockAuthForm({
    onLoginAction,
    onGuestAction
  }: {
    onLoginAction: () => void;
    onGuestAction: () => void;
  }) {
    return (
      <div data-testid="auth-form">
        <button data-testid="login-btn" onClick={onLoginAction}>Login</button>
        <button data-testid="guest-btn" onClick={onGuestAction}>Guest</button>
      </div>
    );
  };
});

jest.mock('../components/Dashboard', () => {
  return function MockDashboard({
    onLogoutAction,
    isGuest
  }: {
    onLogoutAction: () => void;
    isGuest: boolean;
  }) {
    return (
      <div data-testid="dashboard">
        <span data-testid="mode">{isGuest ? 'guest' : 'user'}</span>
        <button data-testid="logout-btn" onClick={onLogoutAction}>Logout</button>
      </div>
    );
  };
});

const mockStorage: Record<string, string> = {};
const mockLocalStorage = {
  getItem: jest.fn((key: string) => mockStorage[key] || null),
  setItem: jest.fn((key: string, value: string) => { mockStorage[key] = value; }),
  removeItem: jest.fn((key: string) => { delete mockStorage[key]; }),
  clear: jest.fn(() => { Object.keys(mockStorage).forEach(k => delete mockStorage[k]); }),
};
Object.defineProperty(window, 'localStorage', { value: mockLocalStorage });

describe('Home (page.tsx)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.clear();
  });

  describe('initial state', () => {
    it('should show Dashboard when token exists in localStorage', async () => {
      mockStorage['token'] = 'existing-token';
      mockLocalStorage.getItem.mockImplementation((key: string) => mockStorage[key] || null);

      render(<Home />);

      await waitFor(() => {
        expect(screen.getByTestId('dashboard')).toBeInTheDocument();
        expect(screen.getByTestId('mode')).toHaveTextContent('user');
        expect(screen.queryByTestId('auth-form')).not.toBeInTheDocument();
      });
    });
  });

  describe('login flow', () => {
    it('should switch from AuthForm to Dashboard after login', async () => {
      const user = userEvent.setup();
      render(<Home />);

      await waitFor(() => expect(screen.getByTestId('auth-form')).toBeInTheDocument());

      await user.click(screen.getByTestId('login-btn'));

      await waitFor(() => {
        expect(screen.getByTestId('dashboard')).toBeInTheDocument();
        expect(screen.getByTestId('mode')).toHaveTextContent('user');
      });
    });
  });

  describe('guest flow', () => {
    it('should switch to Dashboard in guest mode', async () => {
      const user = userEvent.setup();
      render(<Home />);

      await waitFor(() => expect(screen.getByTestId('auth-form')).toBeInTheDocument());

      await user.click(screen.getByTestId('guest-btn'));

      await waitFor(() => {
        expect(screen.getByTestId('dashboard')).toBeInTheDocument();
        expect(screen.getByTestId('mode')).toHaveTextContent('guest');
      });
    });
  });

  describe('logout flow', () => {
    it('should return to AuthForm and clear storage on user logout', async () => {
      const user = userEvent.setup();
      mockStorage['token'] = 'my-token';
      mockStorage['username'] = 'myuser';
      mockLocalStorage.getItem.mockImplementation((key: string) => mockStorage[key] || null);

      render(<Home />);

      await waitFor(() => expect(screen.getByTestId('dashboard')).toBeInTheDocument());

      await user.click(screen.getByTestId('logout-btn'));

      await waitFor(() => {
        expect(screen.getByTestId('auth-form')).toBeInTheDocument();
        expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('token');
        expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('username');
      });
    });

    it('should NOT clear localStorage on guest logout', async () => {
      const user = userEvent.setup();
      render(<Home />);

      await waitFor(() => expect(screen.getByTestId('auth-form')).toBeInTheDocument());
      await user.click(screen.getByTestId('guest-btn'));
      await waitFor(() => expect(screen.getByTestId('dashboard')).toBeInTheDocument());
      mockLocalStorage.removeItem.mockClear();
      await user.click(screen.getByTestId('logout-btn'));
      await waitFor(() => {
        expect(screen.getByTestId('auth-form')).toBeInTheDocument();
        expect(mockLocalStorage.removeItem).not.toHaveBeenCalled();
      });
    });
  });
});