import requests
import json
from typing import List, Dict

def get_categories_with_empty_featured_images() -> List[Dict]:
    """
    Fetches all product categories and filters for those with empty featured_image.
    Handles pagination to get all 300+ categories.
    """
    base_url = "https://dodgerblue-otter-660921.hostingersite.com/wp-json/wp/v2/product-categories"
    categories_with_empty_images = []
    page = 1
    per_page = 100
    total_pages = 1
    
    print("Fetching product categories...")
    
    while page <= total_pages:
        try:
            # Fetch with pagination
            params = {
                'page': page,
                'per_page': per_page
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            categories = response.json()
            
            # Get total pages from headers
            if 'X-WP-TotalPages' in response.headers:
                total_pages = int(response.headers['X-WP-TotalPages'])
            
            # Filter categories with empty featured_image
            for category in categories:
                featured_image = category.get('meta', {}).get('featured_image', '')
                
                if featured_image == '' or featured_image is None or featured_image == 0:
                    categories_with_empty_images.append({
                        'id': category.get('id'),
                        'name': category.get('name'),
                        'slug': category.get('slug'),
                        'featured_image': featured_image
                    })
            
            print(f"Processed page {page}/{total_pages} ({len(categories)} categories)")
            page += 1
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    return categories_with_empty_images

def save_results(categories: List[Dict], filename: str = 'empty_featured_images.json'):
    """Save the results to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {filename}")
    return filename

def print_summary(categories: List[Dict]):
    """Print a summary of the results"""
    print(f"\n{'='*60}")
    print(f"CATEGORIES WITH EMPTY FEATURED IMAGES")
    print(f"{'='*60}")
    print(f"Total categories with empty images: {len(categories)}\n")
    
    for i, cat in enumerate(categories, 1):
        print(f"{i}. ID: {cat['id']:4d} | {cat['name']}")

if __name__ == "__main__":
    # Fetch all categories with empty featured images
    empty_image_categories = get_categories_with_empty_featured_images()
    
    # Print summary
    print_summary(empty_image_categories)
    
    # Save to JSON file
    save_results(empty_image_categories)
    
    # Also save as CSV for easier viewing in spreadsheet apps
    import csv
    csv_filename = 'empty_featured_images.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'slug', 'featured_image'])
        writer.writeheader()
        writer.writerows(empty_image_categories)
    
    print(f"✓ CSV file saved to {csv_filename}")
