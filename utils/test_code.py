import requests
from bs4 import BeautifulSoup


def print_secret_message(url):
    '''
    prints the secret message with the given coordinates in the URL.

    :param url: url which holds the coordinates
    :return: None
    '''
    # Fetch the response fro the url
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the document. Status code: {response.status_code}")
        return

    # Parse the HTML to retrieve the data.
    soup = BeautifulSoup(response.text, 'html.parser')
    rows = soup.find_all('tr')
    grid_data = []

    # Iterate through the table and fetch the coordinates and data.
    for row in rows:
        cols = row.find_all('td')

        if len(cols) == 3:
            x_text = cols[0].get_text(strip=True)
            char = cols[1].get_text(strip=True)
            y_text = cols[2].get_text(strip=True)

            if not x_text.isdigit():
                continue

            x = int(x_text)
            y = int(y_text)

            grid_data.append((x, y, char))

    # Get the maximum dimensions of the grid.
    max_x = max(item[0] for item in grid_data)
    max_y = max(item[1] for item in grid_data)

    # Initialize the 2D grid with empty space.
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Populate the grid and prints the secret code
    for x, y, char in grid_data:
        grid[y][x] = char

    for row in reversed(grid):
        print("".join(row))


if __name__ == "__main__":
    test_url = "https://docs.google.com/document/u/0/d/e/2PACX-1vTMOmshQe8YvaRXi6gEPKKlsC6UpFJSMAk4mQjLm_u1gmHdVVTaeh7nBNFBRlui0sTZ-snGwZM4DBCT/pub?pli=1"
    print_secret_message(test_url)
