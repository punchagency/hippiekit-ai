import sgMail from '@sendgrid/mail';
import { MailDataRequired } from '@sendgrid/mail';

// Initialize SendGrid with API key
sgMail.setApiKey(process.env.SENDGRID_API_KEY || '');

interface SendEmailParams {
  to: string;
  subject: string;
  text: string;
  html?: string;
}

// Helper function to create styled HTML email template
const createEmailTemplate = (subject: string, content: string): string => {
  return `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${subject}</title>
      </head>
      <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0;">
        <div style="background: linear-gradient(135deg, #650084 0%, #8B00C9 100%); padding: 40px 20px; text-align: center;">
          <h1 style="color: white; margin: 0; font-size: 28px; font-weight: bold;">🎨 Hippiekit</h1>
          <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 14px;">Your AI-Powered Shopping Assistant</p>
        </div>
        <div style="background-color: #f8f9fa; padding: 40px 20px;">
          <div style="max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            ${content}
            <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 30px 0;">
            <p style="text-align: center; color: #999; font-size: 12px; margin: 0;">
              © ${new Date().getFullYear()} Hippiekit. All rights reserved.<br>
              <a href="https://hippiekit.com" style="color: #650084; text-decoration: none;">Visit our website</a>
            </p>
          </div>
        </div>
      </body>
    </html>
  `;
};

// Email verification template
export const createVerificationEmail = (verificationLink: string): string => {
  const content = `
    <h2 style="color: #650084; margin-top: 0;">Welcome to Hippiekit! 👋</h2>
    <p>Thank you for signing up. Please verify your email address to get started.</p>
    <p style="margin: 30px 0;">
      <a href="${verificationLink}" style="display: inline-block; padding: 12px 30px; background-color: #650084; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">Verify Email Address</a>
    </p>
    <p style="color: #666; font-size: 13px;">Or copy and paste this link in your browser:</p>
    <p style="color: #999; font-size: 12px; word-break: break-all;">${verificationLink}</p>
    <p style="color: #999; font-size: 13px;">This link will expire in 24 hours.</p>
    <p style="color: #999; font-size: 13px;">If you didn't sign up for Hippiekit, please ignore this email.</p>
  `;
  return createEmailTemplate('Verify Your Email - Hippiekit', content);
};

// OTP email template
export const createOTPEmail = (
  otp: string,
  expiryMinutes: number = 15
): string => {
  const content = `
    <h2 style="color: #650084; margin-top: 0;">Password Reset Request 🔐</h2>
    <p>We received a request to reset your password. Use the code below to proceed:</p>
    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 5px; text-align: center; margin: 30px 0;">
      <p style="margin: 0; color: #999; font-size: 13px;">Your verification code:</p>
      <p style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold; color: #650084; letter-spacing: 5px;">${otp}</p>
    </div>
    <p style="color: #666; font-size: 13px;">This code will expire in ${expiryMinutes} minutes.</p>
    <p style="color: #999; font-size: 13px;">If you didn't request a password reset, please ignore this email and your password will remain unchanged.</p>
  `;
  return createEmailTemplate('Reset Your Password - Hippiekit', content);
};

// Generic email sending function
export async function sendEmail({ to, subject, text, html }: SendEmailParams) {
  try {
    if (!process.env.SENDGRID_API_KEY) {
      throw new Error('SENDGRID_API_KEY is not configured');
    }

    const emailHtml =
      html ||
      createEmailTemplate(
        subject,
        `<p style="white-space: pre-line;">${text}</p>`
      );

    const msg: MailDataRequired = {
      to,
      from: process.env.SENDGRID_FROM_EMAIL || 'developer@punch.cool',
      subject,
      html: emailHtml,
      text,
    };

    const result = await sgMail.send(msg);

    console.log(`📧 Email sent successfully to ${to}`);
    console.log(`   Message ID: ${result[0].headers['x-message-id']}`);

    return {
      success: true,
      messageId: result[0].headers['x-message-id'],
    };
  } catch (error) {
    console.error('❌ Email service error:', error);
    throw new Error(
      `Failed to send email: ${
        error instanceof Error ? error.message : 'Unknown error'
      }`
    );
  }
}

// Send verification email
export async function sendVerificationEmail(
  email: string,
  verificationToken: string,
  baseURL: string = process.env.CLIENT_URL || 'http://localhost:5173'
) {
  const verificationLink = `${baseURL}/verify-email?token=${verificationToken}`;
  const html = createVerificationEmail(verificationLink);

  return sendEmail({
    to: email,
    subject: 'Verify Your Email - Hippiekit',
    text: `Click the link to verify your email: ${verificationLink}`,
    html,
  });
}

// Send OTP email
export async function sendOTPEmail(
  email: string,
  otp: string,
  expiryMinutes: number = 15
) {
  const html = createOTPEmail(otp, expiryMinutes);

  return sendEmail({
    to: email,
    subject: 'Your Password Reset Code - Hippiekit',
    text: `Your password reset code is: ${otp}. This code will expire in ${expiryMinutes} minutes.`,
    html,
  });
}
