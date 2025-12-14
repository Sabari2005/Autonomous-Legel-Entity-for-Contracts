// convertToPd.js
import puppeteer from 'puppeteer';
import fs from 'fs/promises';
import { readFile } from 'fs/promises';

const htmlPath = process.argv[2];
const outputPath = process.argv[3];

const browser = await puppeteer.launch({ headless: 'new' });
const page = await browser.newPage();
const html = await readFile(htmlPath, 'utf-8');

await page.setContent(html, { waitUntil: 'networkidle0' });

await page.pdf({
  path: outputPath,
  format: 'A4',
  printBackground: true,
  margin: {
    top: '70px',
    bottom: '70px',
    left: '70px',
    right: '70px',
  },
});

await browser.close();
