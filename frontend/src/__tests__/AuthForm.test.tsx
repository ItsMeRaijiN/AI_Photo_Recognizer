import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AuthForm from '../components/AuthForm';

const mockPost = jest.fn();
jest.mock('../lib/api', () => ({
  api: {
    post: (...args: unknown[]) => mockPost(...args),
  },
  endpoints: {
    login: '/auth/token',
    register: '/auth/register',
  },
}));

const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
Object.defineProperty(window, 'localStorage', { value: mockLocalStorage });

describe('AuthForm', () => {
  const mockOnLogin = jest.fn();
  const mockOnGuest = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('should render login form by default', () => {
      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);

      expect(screen.getByText('Witaj')).toBeInTheDocument();
      expect(screen.getByText('AI Photo Recognizer')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Login')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Hasło')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Zaloguj się/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Kontynuuj jako Gość/i })).toBeInTheDocument();
    });

    it('should have password input with type="password"', () => {
      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);

      expect(screen.getByPlaceholderText('Hasło')).toHaveAttribute('type', 'password');
    });
  });


  describe('mode switching', () => {
    it('should switch to register mode and back to login', async () => {
      const user = userEvent.setup();
      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);
      expect(screen.getByText('Witaj')).toBeInTheDocument();
      await user.click(screen.getByText(/Nie masz konta\? Załóż je/i));
      expect(screen.getByText('Rejestracja')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Zarejestruj się/i })).toBeInTheDocument();
      await user.click(screen.getByText(/Masz już konto\? Zaloguj się/i));
      expect(screen.getByText('Witaj')).toBeInTheDocument();
    });
  });

  describe('form validation', () => {
    it('should disable submit when fields are empty', () => {
      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);

      expect(screen.getByRole('button', { name: /Zaloguj się/i })).toBeDisabled();
    });

    it('should disable submit when only username is filled', async () => {
      const user = userEvent.setup();
      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);

      await user.type(screen.getByPlaceholderText('Login'), 'testuser');

      expect(screen.getByRole('button', { name: /Zaloguj się/i })).toBeDisabled();
    });

    it('should enable submit when both fields are filled', async () => {
      const user = userEvent.setup();
      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);

      await user.type(screen.getByPlaceholderText('Login'), 'testuser');
      await user.type(screen.getByPlaceholderText('Hasło'), 'testpass');

      expect(screen.getByRole('button', { name: /Zaloguj się/i })).not.toBeDisabled();
    });
  });

  describe('login flow', () => {
    it('should login successfully and store token', async () => {
      const user = userEvent.setup();
      mockPost.mockResolvedValueOnce({ data: { access_token: 'jwt-token-123' } });

      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);

      await user.type(screen.getByPlaceholderText('Login'), 'myuser');
      await user.type(screen.getByPlaceholderText('Hasło'), 'mypass');
      await user.click(screen.getByRole('button', { name: /Zaloguj się/i }));

      await waitFor(() => {
        expect(mockPost).toHaveBeenCalledWith('/auth/token', expect.any(FormData));
        expect(mockLocalStorage.setItem).toHaveBeenCalledWith('token', 'jwt-token-123');
        expect(mockLocalStorage.setItem).toHaveBeenCalledWith('username', 'myuser');
        expect(mockOnLogin).toHaveBeenCalled();
      });
    });

    it('should show error message on login failure', async () => {
      const user = userEvent.setup();
      mockPost.mockRejectedValueOnce({
        response: { data: { detail: 'Nieprawidłowy login lub hasło' } },
      });

      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);

      await user.type(screen.getByPlaceholderText('Login'), 'wrong');
      await user.type(screen.getByPlaceholderText('Hasło'), 'wrong');
      await user.click(screen.getByRole('button', { name: /Zaloguj się/i }));

      await waitFor(() => {
        expect(screen.getByText('Nieprawidłowy login lub hasło')).toBeInTheDocument();
      });
      expect(mockOnLogin).not.toHaveBeenCalled();
    });

    it('should show generic error on network failure', async () => {
      const user = userEvent.setup();
      mockPost.mockRejectedValueOnce(new Error('Network Error'));

      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);

      await user.type(screen.getByPlaceholderText('Login'), 'user');
      await user.type(screen.getByPlaceholderText('Hasło'), 'pass');
      await user.click(screen.getByRole('button', { name: /Zaloguj się/i }));

      await waitFor(() => {
        expect(screen.getByText('Błąd połączenia z serwerem')).toBeInTheDocument();
      });
    });
  });

  describe('registration flow', () => {
    it('should register successfully and switch to login mode', async () => {
      const user = userEvent.setup();
      mockPost.mockResolvedValueOnce({ data: { id: 1, username: 'newuser' } });
      const alertMock = jest.spyOn(window, 'alert').mockImplementation(() => {});
      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);
      await user.click(screen.getByText(/Nie masz konta/i));
      await user.type(screen.getByPlaceholderText('Login'), 'newuser');
      await user.type(screen.getByPlaceholderText('Hasło'), 'newpass');
      await user.click(screen.getByRole('button', { name: /Zarejestruj się/i }));

      await waitFor(() => {
        expect(mockPost).toHaveBeenCalledWith('/auth/register', {
          username: 'newuser',
          password: 'newpass',
        });
        expect(alertMock).toHaveBeenCalledWith('Konto utworzone! Zaloguj się.');
        expect(screen.getByText('Witaj')).toBeInTheDocument();
      });

      alertMock.mockRestore();
    });

    it('should show error when username already exists', async () => {
      const user = userEvent.setup();
      mockPost.mockRejectedValueOnce({
        response: { data: { detail: 'Użytkownik już istnieje' } },
      });

      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);

      await user.click(screen.getByText(/Nie masz konta/i));
      await user.type(screen.getByPlaceholderText('Login'), 'existinguser');
      await user.type(screen.getByPlaceholderText('Hasło'), 'pass');
      await user.click(screen.getByRole('button', { name: /Zarejestruj się/i }));

      await waitFor(() => {
        expect(screen.getByText('Użytkownik już istnieje')).toBeInTheDocument();
      });
    });
  });

  describe('guest mode', () => {
    it('should call onGuestAction without API call', async () => {
      const user = userEvent.setup();
      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);

      await user.click(screen.getByRole('button', { name: /Kontynuuj jako Gość/i }));

      expect(mockOnGuest).toHaveBeenCalled();
      expect(mockPost).not.toHaveBeenCalled();
    });
  });

  describe('loading state', () => {
    it('should disable inputs during login request', async () => {
      const user = userEvent.setup();
      mockPost.mockImplementation(() => new Promise(() => {}));
      render(<AuthForm onLoginAction={mockOnLogin} onGuestAction={mockOnGuest} />);
      await user.type(screen.getByPlaceholderText('Login'), 'user');
      await user.type(screen.getByPlaceholderText('Hasło'), 'pass');
      await user.click(screen.getByRole('button', { name: /Zaloguj się/i }));
      await waitFor(() => {
        expect(screen.getByPlaceholderText('Login')).toBeDisabled();
        expect(screen.getByPlaceholderText('Hasło')).toBeDisabled();
      });
    });
  });
});