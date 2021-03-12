# methods for parsing nested xml with text as text elements like in the bioscope corpus

def get_entries(elm):
    """
    collect tag, text and tail
    """
    text = elm.text
    tail = elm.tail
    return [elm, text, tail]


def dfs(visited, p2c, c2p, node):
    if node not in visited:
        visited.append(node)
        for child in list(node):
            p2c.setdefault(node, []).append(child)

            c2p[child] = node
            dfs(visited, p2c, c2p, child)
    return visited, p2c, c2p


def dfs3(visited, p2c, c2p, node, siblings, i):
    if child_check(visited, p2c, node) is True:
        if node not in visited:

            visited.add(node)
            if node in c2p:
                for sib in p2c[c2p[node]]:
                    if sib not in visited:
                        siblings.setdefault(node, []).append(sib)
                        dfs3(visited, p2c, c2p, sib,  siblings, i)
                if c2p[node] not in visited:
                    dfs3(visited, p2c, c2p, c2p[node], siblings, i)
    else:
        for child in list(node):
            i += 1
            print(i)
            if i > 100: break

            if child not in visited:
                dfs3(visited, p2c, c2p, child,  siblings, i)
    return siblings


def child_check(visited, p2c, node):
    """
    check if child has open children
    :param visited:
    :param p2c:
    :param node:
    :return:
    """
    if node in p2c:
        for child in p2c[node]:
            if child not in visited:
                return False
        return True
    else:
        return True


def get_label(tag):
    if tag.tag == 'xcope':
        return '{}-{}'.format(tag.tag, tag.attrib['id'])
    elif tag.tag == 'cue':
        return '{}-{}-{}'.format(tag.tag, tag.attrib['type'], tag.attrib['ref'])
    else:
        return tag.tag


import itertools


def build_surface(constituents, p2c, c2p, node, siblings, i):
    # check if node has no children or children are already built
    if check_children(constituents, p2c, node):
        if node not in constituents:
            if node in p2c:
                constituents[node] = [(node.text, node)] + list(
                    itertools.chain.from_iterable([constituents[child] for child in p2c[node]])) + [
                                         (node.tail, get_parent_node(node, c2p))]
            else:
                constituents[node] = [(node.text, node), (node.tail, get_parent_node(node, c2p))]
        if node in c2p:
            # check the siblings
            if node in siblings:
                for sib in siblings[node]:
                    if sib not in constituents:
                        build_surface(constituents, p2c, c2p, sib, siblings, i)
            # check the parent
            if c2p[node] not in constituents:
                build_surface(constituents, p2c, c2p, c2p[node], siblings, i)
    else:
        # if children are open
        for child in p2c[node]:
            build_surface(constituents, p2c, c2p, child, siblings, i)
    return constituents


def get_parent_node(node, c2p):
    if node in c2p:
        return c2p[node]
    else:
        return node


def get_all_tags(node, c2p):
    # retrieve tags of the node and all its parents
    tags = [get_label(node)]
    while node in c2p:
        tags.append(get_label(c2p[node]))
        node = c2p[node]
    return tags


def check_children(constituents, p2c, node):
    if node in p2c:
        for child in p2c[node]:
            if child not in constituents:
                return False
        return True
    else:
        return True


def build_surface_ddi(constituents, p2c, c2p, node, siblings, i, xcope_counter, cue_counter):
    # check if node has no children or children are already built
    if node.tag == 'xcope' and 'id' not in node.attrib:
        node.set('id', '{}'.format(xcope_counter))
        xcope_counter += 1

    elif node.tag == 'cue' and 'id' not in node.attrib:
        node.set('id', '{}'.format(cue_counter))
        cue_counter += 1

    if check_children(constituents, p2c, node):
        if node not in constituents:
            if node in p2c:
                constituents[node] = [(node.text, node)] + list(
                    itertools.chain.from_iterable([constituents[child] for child in p2c[node]])) + [
                                         (node.tail, get_parent_node(node, c2p))]
            else:
                constituents[node] = [(node.text, node), (node.tail, get_parent_node(node, c2p))]
        if node in c2p:
            # check the siblings
            if node in siblings:
                for sib in siblings[node]:
                    if sib not in constituents:
                        build_surface_ddi(constituents, p2c, c2p, sib, siblings, i, xcope_counter, cue_counter)
            # check the parent
            if c2p[node] not in constituents:
                build_surface_ddi(constituents, p2c, c2p, c2p[node], siblings, i, xcope_counter, cue_counter)
    else:
        # if children are open
        for child in p2c[node]:
            build_surface_ddi(constituents, p2c, c2p, child, siblings, i, xcope_counter, cue_counter)
    print(constituents)
    return constituents

