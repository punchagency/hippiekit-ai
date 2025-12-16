import express, { Router, Request, Response } from 'express';
import axios from 'axios';

const router: Router = express.Router();

const WP_BASE_URL = 'https://dodgerblue-otter-660921.hostingersite.com/wp-json/wp/v2';

// Proxy WordPress API requests
router.get('/categories', async (req: Request, res: Response) => {
  try {
    const { per_page = '100', slug } = req.query;
    let url = `${WP_BASE_URL}/product-categories?per_page=${per_page}`;
    if (slug) url += `&slug=${slug}`;

    const response = await axios.get(url);
    res.json(response.data);
  } catch (error) {
    console.error('WordPress API error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to fetch categories' 
    });
  }
});

router.get('/products', async (req: Request, res: Response) => {
  try {
    const { page = '1', per_page = '6', search, category_id } = req.query;
    let url = `${WP_BASE_URL}/products?page=${page}&per_page=${per_page}&_embed`;
    
    if (search) url += `&search=${encodeURIComponent(search as string)}`;
    if (category_id) url += `&product-categories=${category_id}`;

    const response = await axios.get(url);
    res.json(response.data);
  } catch (error) {
    console.error('WordPress API error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to fetch products' 
    });
  }
});

router.get('/media/:id', async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const response = await axios.get(`${WP_BASE_URL}/media/${id}`);
    res.json(response.data);
  } catch (error) {
    console.error('WordPress API error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to fetch media' 
    });
  }
});

export default router;
